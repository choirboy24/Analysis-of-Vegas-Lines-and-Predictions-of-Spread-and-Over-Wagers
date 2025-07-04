# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: base
#     language: python
#     name: python3
# ---

# +
# *** Import required libraries

import os
import pandas as pd
import re
import math
import numpy as np
import sklearn
from functions import get_roles, get_favorite, favorite_covered, make_match_key, test_match_key

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
import wandb
# -

# *** Set data paths for each input file
# project_path = os.path.dirname(os.path.abspath(__file__)) <-- won't work in Jupyter, but will in script
# Comment out project_path = os.getcwd() when copying code back to main.py
project_path = os.getcwd()
data_path_strength_sched = os.path.join(
    project_path,
    'data/strength_of_schedule.txt'
)
data_path_2023_2024_lines = os.path.join(
    project_path,
    'data/2023_2024_lines.txt'
)
data_path_merged_cfb_historical_odds = os.path.join(
    project_path,
    'data/merged_cfb_historical_odds.csv'
)
data_path_cfb_box_scores = os.path.join(
    project_path,
    'data/cfb_box_scores_2007_2024.csv'
)
data_path_stad_capacity = os.path.join(
    project_path,
    'data/stadium_capacity.csv'
)

# +
# *** Read in strength_of_schedule.txt to DataFrame
strength_schedule = pd.read_csv(
    data_path_strength_sched, sep='\t', encoding='ansi')
strength = []

# *** Clean school names to make them consistent across all datasets
for i in strength_schedule['Team']:
    i = i.replace('\xa0', '')
    i = re.sub('St$', 'State', i)
    i = re.sub('Car$', 'Carolina', i)
    i = re.sub('So$', 'Southern', i)
    i = re.sub('Mississippi$', 'Ole Miss', i)
    i = re.sub('Louisiana$', 'Louisiana Lafayette', i)
    i = re.sub('UL Monroe', 'Louisiana Monroe', i)
    i = re.sub('UConn', 'Connecticut', i)
    i = re.sub('N Illinois', 'Northern Illinois', i)
    i = re.sub('N Texas', 'North Texas', i)
    i = re.sub('Hawai\'i', 'Hawaii', i)
    i = re.sub('S Florida', 'South Florida', i)
    i = re.sub('W Michigan', 'Western Michigan', i)
    i = re.sub('Middle Tenn', 'Middle Tennessee', i)
    i = re.sub('E Carolina', 'ECU', i)
    i = re.sub('C Michigan', 'Central Michigan', i)
    i = re.sub('E Michigan', 'Eastern Michigan', i)
    i = re.sub('Florida Intl', 'FIU', i)
    i = re.sub('Florida Atlantic', 'FAU', i)
    i = re.sub('W Kentucky', 'Western Kentucky', i)
    i = re.sub('S Alabama', 'South Alabama', i)
    i = re.sub('UMass', 'Massachusetts', i)
    i = re.sub('App State', 'Appalachian State', i)
    i = re.sub('J Madison', 'James Madison', i)
    strength.append(i)
strength_schedule['Team'] = pd.Series(strength)
strength_schedule.drop(['Hi', 'Lo', 'Last'], axis=1, inplace=True)
strength_schedule.rename(columns={'Date': 'full_date', 'Team': 'team', 'Rating': 'rating', 'Season': 'season'}, inplace=True)

# +
# *** Extract wins and losses for each category of strength of schedule (SOS) and put them
# in their own columns
v_1_10_wins = []
v_1_10_losses = []
v_11_25_wins = []
v_11_25_losses = []
v_26_40_wins = []
v_26_40_losses = []

for _, row in strength_schedule.iterrows():
    scores_1_10 = re.findall(r'(\d+)', row['v 1-10'])
    if len(scores_1_10) != 2:
        v_1_10_wins.append(None)
        v_1_10_losses.append(None)
        continue
    v_1_10_win = int(scores_1_10[0])
    v_1_10_loss = int(scores_1_10[1])
    v_1_10_wins.append(v_1_10_win)
    v_1_10_losses.append(v_1_10_loss)

    scores_11_25 = re.findall(r'(\d+)', row['v 11-25'])
    if len(scores_11_25) != 2:
        v_11_25_wins.append(None)
        v_11_25_losses.append(None)
        continue
    v_11_25_win = int(scores_11_25[0])
    v_11_25_loss = int(scores_11_25[1])
    v_11_25_wins.append(v_11_25_win)
    v_11_25_losses.append(v_11_25_loss)

    scores_26_40 = re.findall(r'(\d+)', row['v 26-40'])
    if len(scores_26_40) != 2:
        v_26_40_wins.append(None)
        v_26_40_losses.append(None)
        continue
    v_26_40_win = int(scores_26_40[0])
    v_26_40_loss = int(scores_26_40[1])
    v_26_40_wins.append(v_26_40_win)
    v_26_40_losses.append(v_26_40_loss)
    
# *** Create columns for each category of wins/losses based on SOS
strength_schedule['v_1_10_wins'] = pd.Series(v_1_10_wins)
strength_schedule['v_1_10_losses'] = pd.Series(v_1_10_losses)
strength_schedule['v_11_25_wins'] = pd.Series(v_11_25_wins)
strength_schedule['v_11_25_losses'] = pd.Series(v_11_25_losses)
strength_schedule['v_26_40_wins'] = pd.Series(v_26_40_wins)
strength_schedule['v_26_40_losses'] = pd.Series(v_26_40_losses)

strength_schedule.drop(['v 1-10', 'v 11-25', 'v 26-40'], inplace=True, axis=1)

# +
# *** Read in lines from 2023 and 2024 seasons' odds into DataFrame
stats_23_24 = pd.read_csv(data_path_2023_2024_lines, sep='\t')

teams_list = []

# *** Clean school names to make them consistent across all datasets
for team in stats_23_24['Team']:
    team = team.replace('NIU', 'Northern Illinois')
    team = team.replace('WKU', 'Western Kentucky')
    team = team.replace('Uconn', 'Connecticut')
    team = team.replace('ECU', 'East Carolina')
    teams_list.append(team)
stats_23_24['Team'] = pd.Series(teams_list)
stats_23_24.drop(columns=['Unnamed: 10'], axis=1, inplace=True)
stats_23_24.rename(columns={'Spread': 'spread'}, inplace=True)

# +
# *** Strip out 'vs ' and ' @' from in front of the Opponent teams
# Take each modified team and populate into new columns for home and away teams
home_teams = []
away_teams = []

for index, row in stats_23_24.iterrows():
    opponent = row['Opponent']
    team = row['Team']
    
    if isinstance(opponent, str) and 'vs ' in opponent:
        stats_23_24.at[index, 'Opponent'] = opponent.replace('vs ', '')
        home_teams.append(team)
        away_teams.append(opponent.replace('vs ', ''))
    elif isinstance(opponent, str) and '@ ' in opponent:
        stats_23_24.at[index, 'Opponent'] = opponent.replace('@ ', '')
        home_teams.append(opponent.replace('@ ', ''))
        away_teams.append(team)
    else:
        home_teams.append(None)
        away_teams.append(None)

stats_23_24['home'] = pd.Series(home_teams)
stats_23_24['away'] = pd.Series(away_teams)

# -

# *** Populate list with each unique team in 'Team' column
fbs_teams_list = stats_23_24['Team'].unique().tolist()

# *** Populate teams in the 'away_team' column that are not FBS with the 'FCS' designation
for index, row in stats_23_24.iterrows():
    if row['away'] not in fbs_teams_list:
        stats_23_24.at[index, 'away'] = 'FCS'
    if row['home'] not in fbs_teams_list:
        stats_23_24.at[index, 'home'] = 'FCS'
fcs_teams_stats_away = stats_23_24[stats_23_24['away'] == 'FCS'].index
fcs_teams_stats_home = stats_23_24[stats_23_24['home'] == 'FCS'].index

# *** Drop rows from stats_23_24 DataFrame that have 'FCS' for 'away'
# using the indices of each row that has 'FCS' for 'Opponent'
stats_23_24.drop(fcs_teams_stats_away, inplace=True, errors='ignore')
stats_23_24.drop(fcs_teams_stats_home, inplace=True, errors='ignore')
stats_23_24 = stats_23_24.reset_index(drop=True)

# *** Convert 'full_date' column to date
stats_23_24.rename(columns={'Date': 'full_date'}, inplace=True)
stats_23_24['full_date'] = pd.to_datetime(stats_23_24['full_date'], format='%d-%b-%y')

# *** Give each matchup a 'game_id' consisting of the teams involved and the date
# so as to be able to delete the duplicate matchup
stats_23_24['match_key'] = stats_23_24.apply(
    make_match_key, axis=1
)
stats_23_24['game_id'] = stats_23_24['full_date'].astype(str) + '_' + stats_23_24['match_key']


# *** Drop duplicate matchups
stats_23_24_new = stats_23_24.drop_duplicates(subset='game_id').reset_index(drop=True)

# *** Rename over/under columns
stats_23_24_new.rename(columns={'OU': 'ou', 'Total': 'ou_total'}, inplace=True)

# +
# *** Split out 'Score' column into 'home_score' and 'away_score' columns
home_scores = []
away_scores = []

for _, row in stats_23_24_new.iterrows():
    scores = re.findall(r'(\d+)', row['Score'])
    if len(scores) != 2:
        home_scores.append(None)
        away_scores.append(None)
        continue
    home_score = int(scores[0])
    away_score = int(scores[1])
    if row['Team'] == row['home'] and row['Result'] == 'W':
        home_scores.append(home_score)
        away_scores.append(away_score)
    elif row['Team'] == row['away'] and row['Result'] == 'L':
        home_scores.append(home_score)
        away_scores.append(away_score)
    else:
        home_scores.append(away_score)
        away_scores.append(home_score)
    

stats_23_24_new['home_score'] = pd.Series(home_scores)
stats_23_24_new['away_score'] = pd.Series(away_scores)
# -

# *** Drop unneeded columns
stats_23_24_new.drop(['Team', 'Opponent', 'match_key'], inplace=True, axis=1)

stats_23_24_new.head()

# *** Determine who the favorite/underdog is based on spread value (+/-) and home/away
stats_23_24_new[['fav_side', 'dog_side']] = stats_23_24_new.apply(get_roles, axis=1)

# *** Calculate absolute value of 'spread' value and if the favorite covered the spread or not
stats_23_24_new['spread'] = stats_23_24_new['spread'].abs()
stats_23_24_new['favorite_covered'] = stats_23_24_new.apply(favorite_covered, axis=1)

for i in range(len(stats_23_24_new)):

    c1 = stats_23_24_new.loc[i, 'away_score']
    c2 = stats_23_24_new.loc[i, 'home_score']
    fav_side_home = stats_23_24_new.loc[i, 'fav_side'] == 'home'
    fav_side_away = stats_23_24_new.loc[i, 'fav_side'] == 'away'

    if (c1 + c2) > stats_23_24_new.loc[i, 'ou_total']:
        stats_23_24_new.loc[i, 'ou'] = 'O'
    elif (c1 + c2) < stats_23_24_new.loc[i, 'ou_total']:
        stats_23_24_new.loc[i, 'ou'] = 'U'
    else:
        stats_23_24_new.loc[i, 'ou'] = 'E'
    
    if (c1 > c2) and (fav_side_home):
        stats_23_24_new.loc[i, 'spread_away_cover'] = 'y'
        stats_23_24_new.loc[i, 'spread_home_cover'] = 'n'
    elif (c1 > c2) and (fav_side_away):
        if (c1 - c2) > stats_23_24_new.loc[i, 'spread']:
            stats_23_24_new.loc[i, 'spread_away_cover'] = 'y'
            stats_23_24_new.loc[i, 'spread_home_cover'] = 'n'
        else:
            stats_23_24_new.loc[i, 'spread_away_cover'] = 'n'
            stats_23_24_new.loc[i, 'spread_home_cover'] = 'y'
    elif (c2 > c1) and fav_side_home:
        if (c2 - c1) > stats_23_24_new.loc[i, 'spread']:
            stats_23_24_new.loc[i, 'spread_home_cover'] = 'y'
            stats_23_24_new.loc[i, 'spread_away_cover'] = 'n'
        else:
            stats_23_24_new.loc[i, 'spread_home_cover'] = 'n'
            stats_23_24_new.loc[i, 'spread_away_cover'] = 'y'
    elif (c2 > c1) and fav_side_away:
        if (c2 - c1) > stats_23_24_new.loc[i, 'spread']:
            stats_23_24_new.loc[i, 'spread_home_cover'] = 'y'
            stats_23_24_new.loc[i, 'spread_away_cover'] = 'n'
        else:
            stats_23_24_new.loc[i, 'spread_home_cover'] = 'n'
            stats_23_24_new.loc[i, 'spread_away_cover'] = 'y'
    else:
        stats_23_24_new.loc[i, 'spread_home_cover'] = 'eq'
        stats_23_24_new.loc[i, 'spread_away_cover'] = 'eq'

stats_23_24_new.drop(columns=['Game', 'Result', 'Score', 'ATS'], axis=1, inplace=True)

stats_23_24_new['ml_fav'] = pd.Series(dtype='Int64')
stats_23_24_new['ml_dog'] = pd.Series(dtype='Int64')

# *** Load historical odds DataFrame
hist_odds = pd.read_csv(data_path_merged_cfb_historical_odds)
hist_odds['Season'] = pd.to_numeric(
    hist_odds['Season'], errors='coerce').fillna(0).astype(int)
hist_odds = hist_odds.drop(
    ['1st', '2nd', '3rd', '4th'], axis=1, errors='ignore')

# Clean team names, putting spaces in between words of school name
# correcting Miami Florida and Miami Ohio to Miami FL and Miami OH to
# make it consistent across the other datasets.  Also changing 'St' to 'State'
teams_list = []
for team in hist_odds['Team']:
    team = str(team)
    team = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', team)
    team = team.replace(' U', '')
    team = team.replace('Miami Florida', 'Miami FL')
    team = team.replace('Miami Ohio', 'Miami OH')
    team = team.replace("ULMonroe", "Louisiana Monroe")
    team = team.replace("ULLafayette", "Louisiana Lafayette")
    team = re.sub(r'St$', 'State', team)
    team = re.sub(r'Kent$', 'Kent State', team)
    team = team.replace('So Mississippi', 'Southern Miss')
    teams_list.append(team)
hist_odds['Team'] = pd.Series(teams_list)


# +
# *** Delete contests where one or the other (or both) teams
# have invalid data in them
def mark_bad_pair(df):
    drop_rows = []
    bad_close_vals = ['', 'pk', 'PK', 'nl', 'NL']
    for i in range(0, len(df) - 1, 2):  # step by 2 to process game pairs
        c1 = str(df.loc[i, 'Close']).strip().lower()
        c2 = str(df.loc[i + 1, 'Close']).strip().lower()

        # If either row has an invalid Close value
        if c1 in bad_close_vals or c2 in bad_close_vals or pd.isna(df.loc[i, 'Close']) or pd.isna(df.loc[i + 1, 'Close']):
            drop_rows.extend([i, i + 1])

    return df.drop(index=drop_rows).reset_index(drop=True)

# Apply it
hist_odds = mark_bad_pair(hist_odds)

# +
# *** Combine both rows in a contest in the hist_odds DataFrame into one row

hist_odds['month'] = pd.Series(dtype='Int64')
hist_odds['day'] = pd.Series(dtype='Int64')

hist_odds['fav_side'] = pd.Series(dtype='str')
hist_odds['dog_side'] = pd.Series(dtype='str')

for i in range(0, len(hist_odds) - 1, 2):
    raw_date_str = str(hist_odds.loc[i, 'Date']).zfill(4)
    hist_odds.loc[i, 'raw_date_str'] = raw_date_str
    hist_odds.loc[i, 'month'] = int(raw_date_str[:-2])
    hist_odds.loc[i, 'day'] = int(raw_date_str[-2:])
    hist_odds.loc[i, 'full_date'] = (str(hist_odds.loc[i, 'Season']) + '-' + str(hist_odds.loc[i, 'month']) + '-' + str(hist_odds.loc[i, 'day']))

    hist_odds.loc[i, 'away_score'] = hist_odds.loc[i, 'Final']
    hist_odds.loc[i, 'home_score'] = hist_odds.loc[i + 1, 'Final']

    hist_odds.loc[i, 'away'] = hist_odds.loc[i, 'Team']
    hist_odds.loc[i, 'home'] = hist_odds.loc[i + 1, 'Team']

    c1 = pd.to_numeric(hist_odds.loc[i, 'Close'], errors='coerce')
    c2 = pd.to_numeric(hist_odds.loc[i + 1, 'Close'], errors='coerce')

    c3 = hist_odds.loc[i, 'away_score']
    c4 = hist_odds.loc[i, 'home_score']

    # Only proceed if both c1 and c2 are not NaN
    # If either c1 or c2 is NaN, skip this row
    # *** Apply favorite for home or away to historical odds
    # and combine over/under and spread on to one line depending on which number is higher
    # (the spread number is ALWAYS going to be less than the over/under)
    if not (np.isnan(c1) or np.isnan(c2)):
        if c1 > c2:
            hist_odds.loc[i, 'ou_total'] = c1
            hist_odds.loc[i, 'spread'] = c2
            # Added next 4 rows
            if c2 == 0:
                hist_odds.loc[i, 'fav_side'] = 'none'
                hist_odds.loc[i, 'dog_side'] = 'none'
            else:
                hist_odds.loc[i, 'fav_side'] = 'home'
                hist_odds.loc[i, 'dog_side'] = 'away'
        else:
            hist_odds.loc[i, 'ou_total'] = c2
            hist_odds.loc[i, 'spread'] = c1
            # Added next 4 rows
            if c1 == 0:
                hist_odds.loc[i, 'fav_side'] = 'none'
                hist_odds.loc[i, 'dog_side'] = 'none'
            else:
                hist_odds.loc[i, 'fav_side'] = 'away'
                hist_odds.loc[i, 'dog_side'] = 'home'

        if (c3 + c4) > hist_odds.loc[i, 'ou_total']:
            hist_odds.loc[i, 'ou'] = 'O'
        elif (c3 + c4) < hist_odds.loc[i, 'ou_total']:
            hist_odds.loc[i, 'ou'] = 'U'
        else:
            hist_odds.loc[i, 'ou'] = 'E'
# -

# *** Populate teams in the 'away_team' column that are not FBS with the 'FCS' designation
for index, row in hist_odds.iterrows():
    if row['away'] not in fbs_teams_list:
        hist_odds.at[index, 'away'] = 'FCS'
    if row['home'] not in fbs_teams_list:
        hist_odds.at[index, 'home'] = 'FCS'
fcs_teams_stats_away = hist_odds[hist_odds['away'] == 'FCS'].index
fcs_teams_stats_away = hist_odds[hist_odds['home'] == 'FCS'].index

# *** Drop rows from stats_23_24 DataFrame that have 'FCS' for 'away_team'
# using the indices of each row that has 'FCS' for 'Opponent'
hist_odds.drop(fcs_teams_stats_away, inplace=True, errors='ignore')
hist_odds.drop(fcs_teams_stats_home, inplace=True, errors='ignore')
hist_odds = hist_odds.reset_index(drop=True)

# *** Mark whether the home or away team covered the spread
for i in range(len(hist_odds)):        
    if (c3 > c4) and (hist_odds.loc[i, 'fav_side'] == 'home'):
        hist_odds.loc[i, 'spread_away_cover'] = 'y'
        hist_odds.loc[i, 'spread_home_cover'] = 'n'
    elif (c3 > c4) and (hist_odds.loc[i, 'fav_side'] == 'away'):
        if (c3 - c4) > hist_odds.loc[i, 'spread']:
            hist_odds.loc[i, 'spread_away_cover'] = 'y'
            hist_odds.loc[i, 'spread_home_cover'] = 'n'
        elif (c3 - c4) < hist_odds.loc[i, 'spread']:
            hist_odds.loc[i, 'spread_away_cover'] = 'n'
            hist_odds.loc[i, 'spread_home_cover'] = 'y'
        else:
            hist_odds.loc[i, 'spread_away_cover'] = 'eq'
            hist_odds.loc[i, 'spread_home_cover'] = 'eq'
            
    elif (c4 > c3) and (hist_odds.loc[i, 'fav_side'] == 'home'):
        if (c4 - c3) > hist_odds.loc[i, 'spread']:
            hist_odds.loc[i, 'spread_home_cover'] = 'y'
            hist_odds.loc[i, 'spread_away_cover'] = 'n'
        else:
            hist_odds.loc[i, 'spread_home_cover'] = 'n'
            hist_odds.loc[i, 'spread_away_cover'] = 'y'
    else:
        if (c4 - c3) > hist_odds.loc[i, 'spread']:
            hist_odds.loc[i, 'spread_home_cover'] = 'y'
            hist_odds.loc[i, 'spread_away_cover'] = 'n'
        else:
            hist_odds.loc[i, 'spread_home_cover'] = 'n'
            hist_odds.loc[i, 'spread_away_cover'] = 'y'

hist_odds['ML'] = pd.to_numeric(hist_odds['ML'], errors='coerce')

# +
# *** Determine the money line favorite and underdog
hist_odds['ml_fav'] = pd.Series(dtype='Int64')
hist_odds['ml_dog'] = pd.Series(dtype='Int64')

for i in range(0, len(hist_odds) - 1, 2):
    if (hist_odds.loc[i, 'ML'] != 'NL') and (hist_odds.loc[i+1, 'ML'] != 'NL'):
        if hist_odds.loc[i, 'ML'] < 0:
            hist_odds.loc[i, 'ml_fav'] = hist_odds.loc[i, 'ML']
        else:
            hist_odds.loc[i, 'ml_dog'] = hist_odds.loc[i, 'ML']

        if hist_odds.loc[i+1, 'ML'] < 0:
            hist_odds.loc[i, 'ml_fav'] = hist_odds.loc[i+1, 'ML']
        else:
            hist_odds.loc[i, 'ml_dog'] = hist_odds.loc[i+1, 'ML']
        
hist_odds = hist_odds.dropna(subset=['home', 'away'], axis=0).reset_index(drop=True)
hist_odds = hist_odds.drop(['Rot', 'VH', 'Team', 'Final', 'ML', 'Close'], axis=1, errors='ignore').reset_index(drop=True)
# -

# *** Create match key from teams in contest
hist_odds['match_key'] = hist_odds.apply(make_match_key, axis=1)
hist_odds['game_id'] = hist_odds['full_date'].astype(str) + '_' + hist_odds['match_key']

hist_odds['full_date'] = pd.to_datetime(hist_odds['full_date'], errors="coerce")

hist_odds.columns = [col.lower().strip() for col in hist_odds.columns]

# *** Determine if the favorite covered the spread
hist_odds['favorite_covered'] = hist_odds.apply(favorite_covered, axis=1)

# *** Drop unneedefd columns from historical odds dataset
hist_odds.drop(['season', 'date', 'month', 'day', 'raw_date_str', 'match_key'], axis=1, inplace=True, errors='ignore')

# +
# *** Calculate mean of existing money line favorites and underdogs by spread
# to impute in missing money line fields
# Only include rows where both ml_fav and ml_dog exist
valid_rows = hist_odds[hist_odds['ml_fav'].notna() & hist_odds['ml_dog'].notna()]

spread_means = valid_rows.groupby('spread').agg({
    'ml_fav': 'mean',
    'ml_dog': 'mean'
}).rename(columns={
    'ml_fav': 'mean_fav_ml',
    'ml_dog': 'mean_dog_ml'
})
spread_means['spread'] = spread_means.index
# -

spread_means.dropna(inplace=True, axis=0)

spread_means.loc[0.0, ['mean_fav_ml']] = -110
spread_means.loc[0.0, ['mean_dog_ml']] = -110


# *** Function to impute missing moneylines in historical odds dataset
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


stats_23_24_new = stats_23_24_new.apply(fill_all_mls, axis=1)

hist_odds = hist_odds.apply(fill_missing_mls, axis=1)

stats_23_24_new.dropna(subset=['ml_fav', 'ml_dog'], inplace=True)

hist_odds.dropna(subset=['ml_fav', 'ml_dog'], inplace=True)

stats_23_24_new.isna().any()

hist_odds.isna().any()

ml_conv = spread_means.copy().reset_index(drop=True)

ml_conv.rename(columns={'mean_fav_ml': 'ml_fav', 'mean_dog_ml': 'ml_dog'}, inplace=True)

stats_23_24_new['spread'] = stats_23_24_new['spread'].abs()

# *** Ensure all columns in the 'ml_conv' DataFrame are numeric and that 'fav_ml' and 'dog_ml'
# columns are integers
ml_conv['spread'] = pd.to_numeric(ml_conv['spread'])
ml_conv['ml_fav'] = pd.to_numeric(ml_conv['ml_fav']).round()
ml_conv['ml_dog'] = pd.to_numeric(ml_conv['ml_dog']).round()

ml_conv.dropna().reset_index(drop=True)

# Read in cfb_box_scores_2004_2024.csv to DataFrame
box_scores = pd.read_csv(data_path_cfb_box_scores)
# Drop columns not needed for analysis
box_scores.drop(['rank_away', 'rank_home', 'q1_away', 'q2_away',
                              'q3_away', 'q4_away', 'ot_away', 'q1_home',
                              'q2_home', 'q3_home', 'q4_home', 'ot_home',
                              'tv'], errors='ignore', inplace=True, axis=1)
# Clean school names to make them consistent across all datasets
teams_list = []
for team in box_scores['home']:
    team = team.replace('(OH)', 'OH')
    team = team.replace('(FL)', 'FL')
    team = team.replace('UL-Lafayette', 'Louisiana Lafayette')
    team = team.replace('UL-Monroe', 'Louisiana Monroe')
    team = team.replace('Florida Atlantic', 'FAU')
    team = team.replace('East Carolina', 'ECU')
    team = team.replace('Sam Houston State', 'Sam Houston')
    teams_list.append(team)
box_scores['home'] = teams_list
teams_list = []
for team in box_scores['away']:
    team = team.replace('(OH)', 'OH')
    team = team.replace('(FL)', 'FL')
    team = team.replace('UL-Lafayette', 'Louisiana Lafayette')
    team = team.replace('UL-Monroe', 'Louisiana Monroe')
    team = team.replace('Florida Atlantic', 'FAU')
    team = team.replace('East Carolina', 'ECU')
    team = team.replace('Sam Houston State', 'Sam Houston')
    teams_list.append(team)
box_scores['away'] = teams_list

# *** Convert 'date' column in box_scores DataFrame to date
box_scores['full_date'] = pd.to_datetime(box_scores['full_date'], errors="coerce")

# *** Create match_key column consisting of both teams
box_scores['match_key'] = box_scores.apply(make_match_key, axis=1)
box_scores['game_id'] = box_scores['full_date'].astype(str) + '_' + box_scores['match_key']

# *** Locate non-FBS teams and mark them as 'FCS'
for index, row in box_scores.iterrows():
    if row['away'] not in fbs_teams_list:
        box_scores.at[index, 'away'] = 'FCS'
    if row['home'] not in fbs_teams_list:
        box_scores.at[index, 'home'] = 'FCS'
fcs_teams_stats_away = box_scores[box_scores['away'] == 'FCS'].index
fcs_teams_stats_home = box_scores[box_scores['home'] == 'FCS'].index

# *** Drop rows from stats_23_24 DataFrame that have 'FCS' for 'away_team'
# using the indices of each row that has 'FCS' for 'Opponent'
box_scores.drop(fcs_teams_stats_away, inplace=True, errors='ignore')
box_scores.drop(fcs_teams_stats_home, inplace=True, errors='ignore')
box_scores = box_scores.reset_index(drop=True)

# *** Impute 'n' for games where they are not rivalries
for i in range(0, len(box_scores)):
    if pd.isna(box_scores.loc[i, 'rivalry']):
        box_scores.loc[i, 'rivalry'] = 'n'
        

# *** Load stadium capacity into DataFrame to use for imputation of box_scores['capacity'] column
stad_capacity = pd.read_csv(data_path_stad_capacity)
stad_capacity = stad_capacity.drop('Stadium', axis=1)
stad_capacity.set_index('home', inplace=True)

# +
# *** Impute capacity from stad_capacity into box_scores
for i, row in box_scores.iterrows():
    home_team = row['home']
    capacity = stad_capacity.loc[home_team, 'capacity']
    if pd.isna(box_scores.loc[i, 'capacity']):
        box_scores.at[i, 'capacity'] = capacity

# Convert to 'capacity' to numeric
box_scores['capacity'] = pd.to_numeric(box_scores['capacity'], errors='coerce').astype('Int64')
# -

# *** Calculate mean on attendance for each team
attendance_means = box_scores.groupby('home').agg({
    'attendance': 'mean',
})
attendance_means['home'] = attendance_means.index
attendance_means['attendance'] = attendance_means['attendance'].round()

# *** Impute attendance means for missing games
for i, row in box_scores.iterrows():
    attend_home = row['home']
    attendance = attendance_means.loc[attend_home, 'attendance']
    if pd.isna(box_scores.loc[i, 'attendance']):
        box_scores.at[i, 'attendance'] = attendance

# *** Calculate percentage of attendance/capacity
for i, row in box_scores.iterrows():
    attend_percent = row['attendance'] / row['capacity']
    box_scores['attend_percent']  = attend_percent

stats_23_24_new.sort_values('game_id', axis=0, inplace=True)

hist_odds.sort_values('game_id', axis=0, inplace=True)

box_scores.sort_values('game_id', axis=0, inplace=True)

# Merge stats_23_24 and hist_ods DataFrames
merged_odds_df = pd.concat([hist_odds, stats_23_24_new])

merged_odds_df.set_index(['game_id'], inplace=True, verify_integrity=True)
merged_odds_df.index

box_scores.set_index(['game_id'], inplace=True, verify_integrity=True)

# Merge odds and box scores DataFrames
merged_df = box_scores.merge(merged_odds_df, how='inner', on='game_id')

# Write separate and merged DataFrames to new CSV files
stats_23_24_new.to_csv('data/stats_23_24_new.csv')

hist_odds.to_csv('data/hist_odds.csv')

merged_odds_df.to_csv('data/merged_odds_df.csv')

merged_df.to_csv('data/merged_df.csv')

# %history -f my_history.txt

"""
if __name__ == "__main__":
    # Load the dataset

    # Display the first few rows of the dataset
    print(data.head())

    # Define features and target variable
    X = data.drop('median_house_value', axis=1)
    y = data['median_house_value']

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42)

    # Preprocessing for numerical features
    numeric_features = X.select_dtypes(include=[
    'int64', 'float64']).columns.tolist()
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Preprocessing for categorical features
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Create a pipeline that combines preprocessing
    # with a linear regression model
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'Mean Squared Error: {mse}')
    print(f'R^2 Score: {r2}')
"""
