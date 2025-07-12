# *** Import required libraries

import pandas as pd
from joblib import load


def go(pipeline_spread, pipeline_ou, pipeline_w, merged_df):

    def build_feature_vector(team_home, team_away, game_date, rivalry, spread,
                             ou_total, fav_side, ml_fav, ml_dog, data_source):
        """
        Constructs a model-ready feature vector for a single game.

        Parameters:
            team_home (str): Home team name
            team_away (str): Away team name
            game_date (datetime or str): Date of game for stat cutoff
            spread (float): Spread line
            ou_total (float): Over/under line
            fav_side (str): 'home' or 'away'
            data_source (pd.DataFrame): Historical stats table
            (season averages, matchups, etc.)

        Returns:
            pd.DataFrame: Single-row DataFrame with all engineered features
        """

        prediction_cols = ['full_date', 'away', 'home', 'ou_total', 'spread',
                           'ml_fav', 'ml_dog', 'home_avg', 'away_avg',
                           'home_yards_avg', 'away_yards_avg',
                           'home_pass_yards_avg', 'away_pass_yards_avg',
                           'home_first_downs_avg', 'away_first_downs_avg',
                           'home_third_down_avg', 'away_third_down_avg',
                           'home_fumbles_avg', 'away_fumbles_avg',
                           'home_ints_avg', 'away_ints_avg',
                           'home_pen_yards_avg', 'away_pen_yards_avg',
                           'home_possession_avg', 'away_possession_avg',
                           'rivalry_n', 'rivalry_y', 'fav_side_away',
                           'fav_side_home', 'dog_side_away', 'dog_side_home',
                           'conf_home_aac', 'conf_home_acc', 'conf_home_big10',
                           'conf_home_big12', 'conf_home_cusa',
                           'conf_home_ind', 'conf_home_mac', 'conf_home_mwc',
                           'conf_home_pac12', 'conf_home_sec',
                           'conf_home_sun-belt', 'conf_away_aac',
                           'conf_away_acc', 'conf_away_big10',
                           'conf_away_big12', 'conf_away_cusa',
                           'conf_away_ind', 'conf_away_mac',
                           'conf_away_mwc', 'conf_away_pac12',
                           'conf_away_sec', 'conf_away_sun-belt']

        # Filter for team stats up to the given date
        home_stats = data_source[(data_source['home'] == team_home) &
                                 (data_source['full_date'] < game_date)]
        away_stats = data_source[(data_source['away'] == team_away) &
                                 (data_source['full_date'] < game_date)]

        # Aggregate averages
        home_avg = home_stats.mean(numeric_only=True)
        away_avg = away_stats.mean(numeric_only=True)

        for col, val in home_avg.items():
            if col in prediction_cols:
                home_stats[f'{col}'] = val

        for col, val in away_avg.items():
            if col in prediction_cols:
                away_stats[f'{col}'] = val

        full_stats = pd.concat([home_stats, away_stats], axis=0)
        stats_means = full_stats.mean(numeric_only=True)

        # Build feature dict
        row = {
            'spread': spread,
            'ou_total': ou_total,
            'ml_fav': ml_fav,
            'ml_dog': ml_dog,
            'fav_side_home': 1 if fav_side == 'home' else 0,
            'fav_side_away': 1 if fav_side == 'away' else 0,
            'rivalry_y': 1 if 'rivalry' == 'y' else 0,
            'rivalry_n': 1 if 'rivalry' == 'n' else 0
        }

        row.update(stats_means.to_dict())

        # Convert to DataFrame and fill missing columns as needed
        row_df = pd.DataFrame([row])
        return row_df
    team1 = input("Enter Team 1: ")
    team2 = input("Enter Team 2: ")
    spread = float(input("Enter the spread: "))
    ou = float(input("Enter the over/under (total points): "))
    ml_fav = int(input("Enter the moneyline for the favorite: "))
    ml_dog = int(input("Enter the moneyline for the underdog: "))
    rivalry = input("Is this a rivalry game? (y/n): ").lower()

    predict_game = build_feature_vector(
        team_home=team1,
        team_away=team2,
        game_date=pd.Timestamp.now(),  # You may want to replace this with the actual game date
        rivalry=rivalry,
        spread=spread,
        ou_total=ou,
        fav_side='home' if ml_fav < ml_dog else 'away',  # Example logic, adjust as needed
        ml_fav=ml_fav,
        ml_dog=ml_dog,
        data_source=merged_df
    )

    # Load trained pipeline (already fitted and saved)
    expected_spread_cols = load('models/spread_feature_names.pkl')
    expected_ou_cols = load('models/ou_feature_names.pkl')
    expected_underdog_cols = load('models/w_feature_names.pkl')

    # Align and fill prediction row
    for col in expected_spread_cols:
        if col not in predict_game.columns:
            predict_game[col] = 0

    for col in expected_ou_cols:
        if col not in predict_game.columns:
            predict_game[col] = 0

    for col in expected_underdog_cols:
        if col not in predict_game.columns:
            predict_game[col] = 0

    pipeline_spread = load('models/pipeline_spread.pkl')
    pipeline_ou = load('models/pipeline_ou.pkl')
    pipeline_w = load('models/pipeline_w.pkl')

    # Align input for each model
    def align(row_df, feature_list):
        for col in feature_list:
            if col not in row_df.columns:
                row_df[col] = 0
        return row_df[feature_list]

    champ_spread = align(predict_game, expected_spread_cols)
    champ_ou = align(predict_game, expected_ou_cols)
    champ_underdog = align(predict_game, expected_underdog_cols)

    # Predict
    pred_spread = pipeline_spread.predict(champ_spread)[0]
    pred_spread = 'Favorite covered the spread.' if pred_spread else 'Underdog covered the spread.'
    proba_spread = pipeline_spread.predict_proba(champ_spread)[0]

    pred_ou = pipeline_ou.predict(champ_ou)[0]
    pred_ou = 'The total went over.' if pred_ou else 'The total stayed under.'
    proba_ou = pipeline_ou.predict_proba(champ_ou)[0]

    pred_dog = pipeline_w.predict(champ_underdog)[0]
    pred_dog = 'Underdog won outright.' if pred_dog else 'Favorite secured the victory.'
    proba_dog = pipeline_w.predict_proba(champ_underdog)[0]

    print("\nSpread Prediction:")
    print(f"{pred_spread}")
    print(f"   Confidence — Favorite: {proba_spread[1]:.2%} | Underdog: {proba_spread[0]:.2%}")

    print("\nTotal Points Prediction:")
    print(f"{pred_ou}")
    print(f"   Confidence — Over: {proba_ou[1]:.2%} | Under: {proba_ou[0]:.2%}")

    print("\nUnderdog Win Prediction:")
    print(f"{pred_dog}")
    print(f"   Confidence — Underdog: {proba_dog[1]:.2%} | Favorite: {proba_dog[0]:.2%}")


if __name__ == "__main__":
    go()
