# Import required libraries
import numpy as np
import pandas as pd
from notebook.ipynb import spread_means, merged_df
import matplotlib as plt
from scipy.stats import binom, binomtest

from sklearn.model_selection import train_test_split, cross_validate, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

# *** Delete contests where one or the other (or both) teams
# have invalid data in them
def mark_bad_pair(df):
    drop_rows = []
    bad_close_vals = ['', 'pk', 'PK', 'nl', 'NL']
    for i in range(0, len(df) - 1, 2):  # step by 2 to process game pairs
        c1 = str(df.loc[i, 'Close']).strip().lower()
        c2 = str(df.loc[i + 1, 'Close']).strip().lower()

        # If either row has an invalid Close value
        if (
            c1 in bad_close_vals
            or c2 in bad_close_vals
            or pd.isna(df.loc[i, 'Close'])
            or pd.isna(df.loc[i + 1, 'Close'])
        ):
            drop_rows.extend([i, i + 1])

    return df.drop(index=drop_rows).reset_index(drop=True)


# Determine whether the home or away team is the favorite bassed on the spread
def get_roles(row):
    if row['spread'] < 0:
        return pd.Series({'fav_side': 'home', 'dog_side': 'away'})
    elif row['spread'] > 0:
        return pd.Series({'fav_side': 'away', 'dog_side': 'home'})
    else:
        return pd.Series({'fav_side': 'none', 'dog_side': 'none'})  # pick'em


# *** Function definition to determine if the home or away team
# is the betting favorite
def get_favorite(row):
    if row['spread'] < 0:
        return row['home']
    elif row['spread'] > 0:
        return row['away']
    else:
        return row['none']


# *** Function to determine if the favorite covered the spread
def favorite_covered(row):
    if row['fav_side'] == 'home':
        margin = row['home_score'] - row['away_score']
        return 'y' if margin > abs(row['spread']) else 'n'
    elif row['fav_side'] == 'away':
        margin = row['away_score'] - row['home_score']
        return 'y' if margin > abs(row['spread']) else 'n'
    else:
        return 'eq'  # Handle pick'em or edge cases

def make_match_key(row):
    teams = [str(row['home']), str(row['away'])]
    teams = [team.strip() for team in teams if team and team != 'nan']
    return '_'.join(sorted(teams))

# Function to impute missing moneylines in historical odds dataset
def fill_missing_mls(row):
    spread_val = row['spread']
    if pd.isna(spread_val) or spread_val not in spread_means.index:
        return row  # No lookup if spread is missing or not in index

    means = spread_means.loc[spread_val]

    if pd.isna(row['ml_fav']):
        row['ml_fav'] = means['mean_fav_ml']
    if pd.isna(row['ml_dog']):
        row['ml_dog'] = means['mean_dog_ml']

    return row

# *** Function to impute all moneyline values for 2023-24 odds dataset
def fill_all_mls(row):
    spread_val = row['spread']
    if pd.isna(spread_val) or spread_val not in spread_means.index:
        return row  # No lookup if spread is missing or not in index
    means = spread_means.loc[spread_val]

    if pd.isna(row['ml_fav']):
        row['ml_fav'] = means['mean_fav_ml']
    if pd.isna(row['ml_dog']):
        row['ml_dog'] = means['mean_dog_ml']

    return row

def binomial_validation(k, n, model_name, y_pred, y_true, p_null, alpha=0.05):
    result = binomtest(k, n, p=p_null, alternative='greater')
    print(f"{model_name} Model:")
    print(f"  k = {k}, n = {n}")
    print(f"  p-value = {result.pvalue:.4f}")
    print("  Result:", "Significant" if result.pvalue < alpha else "Not significant")
    print("Result - Test Statistic:", result.statistic)
    print("Result - Confidence Interval:", result.proportion_ci)

def build_feature_vector(team_home, team_away, game_date, rivalry, spread, ou_total, fav_side, ml_fav, ml_dog, data_source):
    """
    Constructs a model-ready feature vector for a single game.

    Parameters:
        team_home (str): Home team name
        team_away (str): Away team name
        game_date (datetime or str): Date of game for stat cutoff
        spread (float): Spread line
        ou_total (float): Over/under line
        fav_side (str): 'home' or 'away'
        data_source (pd.DataFrame): Historical stats table (season averages, matchups, etc.)

    Returns:
        pd.DataFrame: Single-row DataFrame with all engineered features
    """

    excluded_cols = ['season', 'week', 'time_et', 'game_type', 'neutral', 'spread_away_cover', 'spread_home_cover', 'home', 'away', 'full_date', 'game_id', 'conf_home', 'conf_away', 'rivalry', 'fav_side', 'dog_side']

    # Filter for team stats up to the given date
    home_stats = data_source[(data_source['home'] == team_home) & (data_source['full_date'] < game_date)]
    away_stats = data_source[(data_source['away'] == team_away) & (data_source['full_date'] < game_date)]

    # Aggregate averages
    home_avg = home_stats.mean(numeric_only=True)
    away_avg = away_stats.mean(numeric_only=True)

    for col, val in home_avg.items():
        if col not in excluded_cols:
            home_stats[f'{col}'] = val
    
    for col, val in away_avg.items():
        if col not in excluded_cols:
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

def plot_binomial_pmf(n, p, model_name="Model", zoom_ratio=0.1):
    """
    Plots a zoomed-in binomial PMF with annotated expected value.
    
    Parameters:
    - n: Number of trials
    - p: Probability of success per trial
    - model_name: Title label for the model
    - zoom_ratio: Percent of n to use for zoom window (default = 10%)
    """
    
    expected_successes = n * p
    zoom_window = int(n * zoom_ratio)
    
    # Define zoomed range
    r_values = list(range(int(expected_successes - zoom_window), int(expected_successes + zoom_window)))
    dist = [binom.pmf(r, n, p) for r in r_values]

    # Plot
    plt.figure(figsize=(10,6))
    plt.bar(r_values, dist, color='steelblue', edgecolor='black')

    # Add annotation for expected successes
    plt.axvline(expected_successes, color='purple', linestyle='-', linewidth=2,
                label=f'Expected Successes: {expected_successes:.1f}')

    plt.title(f'{model_name} Binomial PMF\nn = {n}, p = {p:.3f}')
    plt.xlabel('Number of Successes')
    plt.ylabel('Probability')
    plt.legend()
    plt.tight_layout()
    plt.show()

    def evaluate_pca_model(X_train, X_test, y_train, y_test, n_components_list):
        scores = []
        for n in n_components_list:
            pipe = Pipeline([
                ('scaler', StandardScaler()),
                ('pca', PCA(n_components=n)),
                ('model', LogisticRegression(max_iter=500, random_state=42))
            ])

            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            scores.append((n, acc))
            print(f"Components: {n:2d} | Accuracy: {acc:.4f}")
        return scores

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

def get_team_past_games_rushes(team, current_date):
    rows_before_date = merged_df[merged_df['full_date'] < current_date]
    home_games = rows_before_date[rows_before_date['home'] == team]
    away_games = rows_before_date[rows_before_date['away'] == team]
    if len(home_games) == 0 and len(away_games) == 0:
        return 0.0   
    home_rush_att_avg = home_games['rush_att_home'].mean()
    away_rush_att_avg = away_games['rush_att_away'].mean()
        
    return home_rush_att_avg, away_rush_att_avg

def get_team_past_games_rush_yards(team, current_date):
    rows_before_date = merged_df[merged_df['full_date'] < current_date]
    home_games = rows_before_date[rows_before_date['home'] == team]
    away_games = rows_before_date[rows_before_date['away'] == team]
    if len(home_games) == 0 and len(away_games) == 0:
        return 0.0   
    home_rush_yards_avg = home_games['rush_yards_home'].mean()
    away_rush_yards_avg = away_games['rush_yards_away'].mean()
        
    return home_rush_yards_avg, away_rush_yards_avg

def get_team_past_games_pass_comp(team, current_date):
    rows_before_date = merged_df[merged_df['full_date'] < current_date]
    home_games = rows_before_date[rows_before_date['home'] == team]
    away_games = rows_before_date[rows_before_date['away'] == team]
    if len(home_games) == 0 and len(away_games) == 0:
        return 0.0   
    home_pass_comp_avg = home_games['pass_comp_home'].mean()
    away_pass_comp_avg = away_games['pass_comp_away'].mean()
        
    return home_pass_comp_avg, away_pass_comp_avg

def get_team_past_games_pass_att(team, current_date):
    rows_before_date = merged_df[merged_df['full_date'] < current_date]
    home_games = rows_before_date[rows_before_date['home'] == team]
    away_games = rows_before_date[rows_before_date['away'] == team]
    if len(home_games) == 0 and len(away_games) == 0:
        return 0.0   
    home_pass_att_avg = home_games['pass_att_home'].mean()
    away_pass_att_avg = away_games['pass_att_away'].mean()
        
    return home_pass_att_avg, away_pass_att_avg

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

def get_team_past_games_third_down_att(team, current_date):
    rows_before_date = merged_df[merged_df['full_date'] < current_date]
    home_games = rows_before_date[rows_before_date['home'] == team]
    away_games = rows_before_date[rows_before_date['away'] == team]
    if len(home_games) == 0 and len(away_games) == 0:
        return 0.0   
    home_third_down_att_avg = home_games['third_down_att_home'].mean()
    away_third_down_att_avg = away_games['third_down_att_away'].mean()
        
    return home_third_down_att_avg, away_third_down_att_avg

def get_team_past_games_fourth_down(team, current_date):
    rows_before_date = merged_df[merged_df['full_date'] < current_date]
    home_games = rows_before_date[rows_before_date['home'] == team]
    away_games = rows_before_date[rows_before_date['away'] == team]
    if len(home_games) == 0 and len(away_games) == 0:
        return 0.0   
    home_fourth_down_avg = home_games['fourth_down_comp_home'].mean()
    away_fourth_down_avg = away_games['fourth_down_comp_away'].mean()
        
    return home_fourth_down_avg, away_fourth_down_avg

def get_team_past_games_fourth_down_att(team, current_date):
    rows_before_date = merged_df[merged_df['full_date'] < current_date]
    home_games = rows_before_date[rows_before_date['home'] == team]
    away_games = rows_before_date[rows_before_date['away'] == team]
    if len(home_games) == 0 and len(away_games) == 0:
        return 0.0   
    home_fourth_down_att_avg = home_games['fourth_down_att_home'].mean()
    away_fourth_down_att_avg = away_games['fourth_down_att_away'].mean()
        
    return home_fourth_down_att_avg, away_fourth_down_att_avg

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

def get_team_past_games_pen(team, current_date):
    rows_before_date = merged_df[merged_df['full_date'] < current_date]
    home_games = rows_before_date[rows_before_date['home'] == team]
    away_games = rows_before_date[rows_before_date['away'] == team]
    if len(home_games) == 0 and len(away_games) == 0:
        return 0.0   
    home_pen_num_avg = home_games['pen_num_home'].mean()
    away_pen_num_avg = away_games['pen_num_away'].mean()
        
    return home_pen_num_avg, away_pen_num_avg

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