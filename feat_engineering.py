# *** Import required libraries

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def go(merged_df):
    merged_df['dog_wins'] = pd.Series(dtype='Int64')
    merged_df['dog_wins'] = (
        ((merged_df['dog_side'] == 'home') &
            (merged_df['score_home'] > merged_df['score_away'])) |
        ((merged_df['dog_side'] == 'away') &
            (merged_df['score_home'] < merged_df['score_away']))
    ).astype(int)

    merged_df.sort_values('full_date', inplace=True, axis=0)

    def get_team_past_games(team, current_date):
        rows_before_date = merged_df[merged_df['full_date'] < current_date]
        home_games = rows_before_date[rows_before_date['home'] == team]
        away_games = rows_before_date[rows_before_date['away'] == team]
        if len(home_games) == 0 and len(away_games) == 0:
            return 0.0
        home_avg = home_games['score_home'].mean()
        away_avg = away_games['score_away'].mean()

        return home_avg, away_avg

    def get_team_past_games_yards(team, current_date):
        rows_before_date = merged_df[merged_df['full_date'] < current_date]
        home_games = rows_before_date[rows_before_date['home'] == team]
        away_games = rows_before_date[rows_before_date['away'] == team]
        if len(home_games) == 0 and len(away_games) == 0:
            return 0.0
        home_yards_avg = home_games['total_yards_home'].mean()
        away_yards_avg = away_games['total_yards_away'].mean()

        return home_yards_avg, away_yards_avg

    def get_team_past_games_rush_yards(team, current_date):
        rows_before_date = merged_df[merged_df['full_date'] < current_date]
        home_games = rows_before_date[rows_before_date['home'] == team]
        away_games = rows_before_date[rows_before_date['away'] == team]
        if len(home_games) == 0 and len(away_games) == 0:
            return 0.0
        home_rush_yards_avg = home_games['rush_yards_home'].mean()
        away_rush_yards_avg = away_games['rush_yards_away'].mean()

        return home_rush_yards_avg, away_rush_yards_avg

    def get_team_past_games_pass_yards(team, current_date):
        rows_before_date = merged_df[merged_df['full_date'] < current_date]
        home_games = rows_before_date[rows_before_date['home'] == team]
        away_games = rows_before_date[rows_before_date['away'] == team]
        if len(home_games) == 0 and len(away_games) == 0:
            return 0.0
        home_pass_yards_avg = home_games['pass_yards_home'].mean()
        away_pass_yards_avg = away_games['pass_yards_away'].mean()

        return home_pass_yards_avg, away_pass_yards_avg

    def get_team_past_games_first_downs(team, current_date):
        rows_before_date = merged_df[merged_df['full_date'] < current_date]
        home_games = rows_before_date[rows_before_date['home'] == team]
        away_games = rows_before_date[rows_before_date['away'] == team]
        if len(home_games) == 0 and len(away_games) == 0:
            return 0.0
        home_first_downs_avg = home_games['first_downs_home'].mean()
        away_first_downs_avg = away_games['first_downs_away'].mean()

        return home_first_downs_avg, away_first_downs_avg

    def get_team_past_games_third_down(team, current_date):
        rows_before_date = merged_df[merged_df['full_date'] < current_date]
        home_games = rows_before_date[rows_before_date['home'] == team]
        away_games = rows_before_date[rows_before_date['away'] == team]
        if len(home_games) == 0 and len(away_games) == 0:
            return 0.0
        home_third_down_avg = home_games['third_down_comp_home'].mean()
        away_third_down_avg = away_games['third_down_comp_away'].mean()

        return home_third_down_avg, away_third_down_avg

    def get_team_past_games_fumbles(team, current_date):
        rows_before_date = merged_df[merged_df['full_date'] < current_date]
        home_games = rows_before_date[rows_before_date['home'] == team]
        away_games = rows_before_date[rows_before_date['away'] == team]
        if len(home_games) == 0 and len(away_games) == 0:
            return 0.0
        home_fumbles_avg = home_games['fum_home'].mean()
        away_fumbles_avg = away_games['fum_away'].mean()

        return home_fumbles_avg, away_fumbles_avg

    def get_team_past_games_ints(team, current_date):
        rows_before_date = merged_df[merged_df['full_date'] < current_date]
        home_games = rows_before_date[rows_before_date['home'] == team]
        away_games = rows_before_date[rows_before_date['away'] == team]
        if len(home_games) == 0 and len(away_games) == 0:
            return 0.0
        home_ints_avg = home_games['int_home'].mean()
        away_ints_avg = away_games['int_away'].mean()

        return home_ints_avg, away_ints_avg

    def get_team_past_games_pen_yards(team, current_date):
        rows_before_date = merged_df[merged_df['full_date'] < current_date]
        home_games = rows_before_date[rows_before_date['home'] == team]
        away_games = rows_before_date[rows_before_date['away'] == team]
        if len(home_games) == 0 and len(away_games) == 0:
            return 0.0
        home_pen_yards_avg = home_games['pen_yards_home'].mean()
        away_pen_yards_avg = away_games['pen_yards_away'].mean()

        return home_pen_yards_avg, away_pen_yards_avg

    def get_team_past_games_possession(team, current_date):
        rows_before_date = merged_df[merged_df['full_date'] < current_date]
        home_games = rows_before_date[rows_before_date['home'] == team]
        away_games = rows_before_date[rows_before_date['away'] == team]
        if len(home_games) == 0 and len(away_games) == 0:
            return 0.0
        home_possession_avg = home_games['possession_home'].mean()
        away_possession_avg = away_games['possession_away'].mean()

        return home_possession_avg, away_possession_avg

    # Add columns for averages if they don't exist
    if 'home_avg' not in merged_df.columns:
        merged_df['home_avg'] = np.nan
    if 'away_avg' not in merged_df.columns:
        merged_df['away_avg'] = np.nan

    for i, row in merged_df.iterrows():
        home_team = row['home']
        away_team = row['away']
        current_date = row['full_date']

        result = get_team_past_games(home_team, current_date)
        if isinstance(result, tuple):
            home_avg, _ = result
        else:
            home_avg = result

        result = get_team_past_games(away_team, current_date)
        if isinstance(result, tuple):
            _, away_avg = result
        else:
            away_avg = result

        merged_df.at[i, 'home_avg'] = home_avg
        merged_df.at[i, 'away_avg'] = away_avg

        # Add columns for averages if they don't exist
    if 'home_yards_avg' not in merged_df.columns:
        merged_df['home_yards_avg'] = np.nan
    if 'away_yards_avg' not in merged_df.columns:
        merged_df['away_yards_avg'] = np.nan

    for i, row in merged_df.iterrows():
        home_team = row['home']
        away_team = row['away']
        current_date = row['full_date']

        result = get_team_past_games_yards(home_team, current_date)
        if isinstance(result, tuple):
            home_yards_avg, _ = result
        else:
            home_yards_avg = result

        result = get_team_past_games_yards(away_team, current_date)
        if isinstance(result, tuple):
            _, away_yards_avg = result
        else:
            away_yards_avg = result

        merged_df.at[i, 'home_yards_avg'] = home_yards_avg
        merged_df.at[i, 'away_yards_avg'] = away_yards_avg

        # Add columns for averages if they don't exist
    if 'home_rush_yards_avg' not in merged_df.columns:
        merged_df['home_avg'] = np.nan
    if 'away_rush_yards_avg' not in merged_df.columns:
        merged_df['away_avg'] = np.nan

    for i, row in merged_df.iterrows():
        home_team = row['home']
        away_team = row['away']
        current_date = row['full_date']

        result = get_team_past_games_rush_yards(home_team, current_date)
        if isinstance(result, tuple):
            home_rush_yards_avg, _ = result
        else:
            home_rush_yards_avg = result

        result = get_team_past_games_rush_yards(away_team, current_date)
        if isinstance(result, tuple):
            _, away_rush_yards_avg = result
        else:
            away_rush_yards_avg = result

        merged_df.at[i, 'home_rush_yards_avg'] = home_rush_yards_avg
        merged_df.at[i, 'away_rush_yards_avg'] = away_rush_yards_avg

    # Add columns for averages if they don't exist
    if 'home_pass_yards_avg' not in merged_df.columns:
        merged_df['home_pass_yards_avg'] = np.nan
    if 'away_pass_yards_avg' not in merged_df.columns:
        merged_df['away_pass_yards_avg'] = np.nan

    for i, row in merged_df.iterrows():
        home_team = row['home']
        away_team = row['away']
        current_date = row['full_date']

        result = get_team_past_games_pass_yards(home_team, current_date)
        if isinstance(result, tuple):
            home_pass_yards_avg, _ = result
        else:
            home_pass_yards_avg = result

        result = get_team_past_games_pass_yards(away_team, current_date)
        if isinstance(result, tuple):
            _, away_pass_yards_avg = result
        else:
            away_pass_yards_avg = result

        merged_df.at[i, 'home_pass_yards_avg'] = home_pass_yards_avg
        merged_df.at[i, 'away_pass_yards_avg'] = away_pass_yards_avg

        # Add columns for averages if they don't exist
    if 'home_first_downs_avg' not in merged_df.columns:
        merged_df['home_first_downs_avg'] = np.nan
    if 'away_first_downs_avg' not in merged_df.columns:
        merged_df['away_first_downs_avg'] = np.nan

    for i, row in merged_df.iterrows():
        home_team = row['home']
        away_team = row['away']
        current_date = row['full_date']

        result = get_team_past_games_first_downs(home_team, current_date)
        if isinstance(result, tuple):
            home_first_downs_avg, _ = result
        else:
            home_first_downs_avg = result

        result = get_team_past_games_first_downs(away_team, current_date)
        if isinstance(result, tuple):
            _, away_first_downs_avg = result
        else:
            away_first_downs_avg = result

        merged_df.at[i, 'home_first_downs_avg'] = home_first_downs_avg
        merged_df.at[i, 'away_first_downs_avg'] = away_first_downs_avg

        # Add columns for averages if they don't exist
    if 'home_third_down_avg' not in merged_df.columns:
        merged_df['home_third_down_avg'] = np.nan
    if 'away_third_down_avg' not in merged_df.columns:
        merged_df['away_third_down_avg'] = np.nan

    for i, row in merged_df.iterrows():
        home_team = row['home']
        away_team = row['away']
        current_date = row['full_date']

        result = get_team_past_games_third_down(home_team, current_date)
        if isinstance(result, tuple):
            home_third_down_avg, _ = result
        else:
            home_third_down_avg = result

        result = get_team_past_games_third_down(away_team, current_date)
        if isinstance(result, tuple):
            _, away_third_down_avg = result
        else:
            away_third_down_avg = result

        merged_df.at[i, 'home_third_down_avg'] = home_third_down_avg
        merged_df.at[i, 'away_third_down_avg'] = away_third_down_avg

        # Add columns for averages if they don't exist
    if 'home_fumbles_avg' not in merged_df.columns:
        merged_df['home_fumbles_avg'] = np.nan
    if 'away_fumbles_avg' not in merged_df.columns:
        merged_df['away_fumbles_avg'] = np.nan

    for i, row in merged_df.iterrows():
        home_team = row['home']
        away_team = row['away']
        current_date = row['full_date']

        result = get_team_past_games_fumbles(home_team, current_date)
        if isinstance(result, tuple):
            home_fumbles_avg, _ = result
        else:
            home_fumbles_avg = result

        result = get_team_past_games_fumbles(away_team, current_date)
        if isinstance(result, tuple):
            _, away_fumbles_avg = result
        else:
            away_fumbles_avg = result

        merged_df.at[i, 'home_fumbles_avg'] = home_fumbles_avg
        merged_df.at[i, 'away_fumbles_avg'] = away_fumbles_avg

        # Add columns for averages if they don't exist
    if 'home_ints_avg' not in merged_df.columns:
        merged_df['home_ints_avg'] = np.nan
    if 'away_ints_avg' not in merged_df.columns:
        merged_df['away_ints_avg'] = np.nan

    for i, row in merged_df.iterrows():
        home_team = row['home']
        away_team = row['away']
        current_date = row['full_date']

        result = get_team_past_games_ints(home_team, current_date)
        if isinstance(result, tuple):
            home_ints_avg, _ = result
        else:
            home_ints_avg = result

        result = get_team_past_games_ints(away_team, current_date)
        if isinstance(result, tuple):
            _, away_ints_avg = result
        else:
            away_ints_avg = result

        merged_df.at[i, 'home_ints_avg'] = home_ints_avg
        merged_df.at[i, 'away_ints_avg'] = away_ints_avg

        # Add columns for averages if they don't exist
    if 'home_pen_yards_avg' not in merged_df.columns:
        merged_df['home_pen_yards_avg'] = np.nan
    if 'away_pen_yards_avg' not in merged_df.columns:
        merged_df['away_pen_yards_avg'] = np.nan

    for i, row in merged_df.iterrows():
        home_team = row['home']
        away_team = row['away']
        current_date = row['full_date']

        result = get_team_past_games_pen_yards(home_team, current_date)
        if isinstance(result, tuple):
            home_pen_yards_avg, _ = result
        else:
            home_pen_yards_avg = result

        result = get_team_past_games_pen_yards(away_team, current_date)
        if isinstance(result, tuple):
            _, away_pen_yards_avg = result
        else:
            away_pen_yards_avg = result

        merged_df.at[i, 'home_pen_yards_avg'] = home_pen_yards_avg
        merged_df.at[i, 'away_pen_yards_avg'] = away_pen_yards_avg

        # Add columns for averages if they don't exist
    if 'home_possession_avg' not in merged_df.columns:
        merged_df['home_possession_avg'] = np.nan
    if 'away_possession_avg' not in merged_df.columns:
        merged_df['away_possession_avg'] = np.nan

    for i, row in merged_df.iterrows():
        home_team = row['home']
        away_team = row['away']
        current_date = row['full_date']

        result = get_team_past_games_possession(home_team, current_date)
        if isinstance(result, tuple):
            home_possession_avg, _ = result
        else:
            home_possession_avg = result

        result = get_team_past_games_possession(away_team, current_date)
        if isinstance(result, tuple):
            _, away_possession_avg = result
        else:
            away_possession_avg = result

        merged_df.at[i, 'home_possession_avg'] = home_possession_avg
        merged_df.at[i, 'away_possession_avg'] = away_possession_avg

    labels_df = pd.get_dummies(
        merged_df[['ou', 'favorite_covered', 'dog_wins']])
    merged_df.drop(columns=[
        'ou',
        'favorite_covered',
        'dog_wins'],
        inplace=True
        )

    df_dummies = merged_df[[
        'rivalry',
        'fav_side',
        'dog_side',
        'conf_home',
        'conf_away'
        ]]
    df_converted = pd.get_dummies(
        df_dummies,
        prefix=None,
        dummy_na=False,
        columns=['rivalry', 'fav_side', 'dog_side', 'conf_home', 'conf_away'],
        sparse=False,
        drop_first=False,
        dtype='int64'
    )

    df_converted['game_id'] = merged_df['game_id']

    merged_df = merged_df.merge(df_converted, how='inner', on='game_id')

    # Train/test split for spread covered
    y_spread = labels_df['favorite_covered_y']
    y_ou = labels_df['ou_O']
    y_w = labels_df['dog_wins']

    merged_df.drop(columns='index', inplace=True, axis=1, errors='ignore')

    numeric_cols = ['ou_total', 'spread', 'ml_fav',
                    'ml_dog', 'home_avg', 'away_avg', 'home_yards_avg',
                    'away_yards_avg', 'home_pass_yards_avg',
                    'away_pass_yards_avg', 'home_first_downs_avg',
                    'away_first_downs_avg', 'home_third_down_avg',
                    'away_third_down_avg', 'home_fumbles_avg',
                    'away_fumbles_avg', 'home_ints_avg', 'away_ints_avg',
                    'home_pen_yards_avg', 'away_pen_yards_avg',
                    'home_possession_avg', 'away_possession_avg',
                    'rivalry_n', 'rivalry_y', 'fav_side_away', 'fav_side_home',
                    'dog_side_away', 'dog_side_home', 'conf_home_aac',
                    'conf_home_acc', 'conf_home_big10', 'conf_home_big12',
                    'conf_home_cusa', 'conf_home_ind', 'conf_home_mac',
                    'conf_home_mwc', 'conf_home_pac12', 'conf_home_sec',
                    'conf_home_sun-belt', 'conf_away_aac', 'conf_away_acc',
                    'conf_away_big10', 'conf_away_big12', 'conf_away_cusa',
                    'conf_away_ind', 'conf_away_mac', 'conf_away_mwc',
                    'conf_away_pac12', 'conf_away_sec', 'conf_away_sun-belt'
                    ]
    X = merged_df[numeric_cols]

    merged_df_copy = merged_df.copy()

    merged_df_copy.to_csv('data/csv-clean/merged_df_copy.csv')

    X.fillna(0, inplace=True)

    # Train/test split (
    # spread for point spread, ou for Over/Under, w for underdog wins)
    (
        X_train, X_test,
        y_train_spread, y_test_spread,
        y_train_ou, y_test_ou,
        y_train_w, y_test_w
    ) = train_test_split(
        X, y_spread, y_ou, y_w,
        test_size=0.2,
        random_state=42
    )

    return (
        X_train, X_test,
        y_train_spread, y_test_spread,
        y_train_ou, y_test_ou,
        y_train_w, y_test_w,
        merged_df
    )


if __name__ == "__main__":
    go()
