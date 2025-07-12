# *** Import required libraries

import pandas as pd
import re
import numpy as np
from functions import (get_roles,
                       mark_bad_pair,
                       favorite_covered,
                       make_match_key
                       )
from eda import stats_23_24, hist_odds, box_scores, stad_capacity


def go(stats_23_24, hist_odds, box_scores, stad_capacity):

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

    # *** Strip out 'vs ' and ' @' from in front of the Opponent teams
    # Take each modified team and populate into new columns for
    # home and away teams
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

    # *** Populate list with each unique team in 'Team' column
    fbs_teams_list = stats_23_24['Team'].unique().tolist()

    # Populate teams in the 'away_team' column that are not FBS with the
    # 'FCS' designation
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
    stats_23_24.reset_index(drop=True, inplace=True)

    # *** Convert 'full_date' column to date
    stats_23_24.rename(columns={'Date': 'full_date'}, inplace=True)
    stats_23_24['full_date'] = pd.to_datetime(stats_23_24['full_date'],
                                              format='%d-%b-%y')

    # *** Give each matchup a 'game_id' consisting of the teams involved
    #  and the date so as to be able to delete the duplicate matchup
    stats_23_24['match_key'] = stats_23_24.apply(
        make_match_key, axis=1
    )
    stats_23_24['game_id'] = stats_23_24['full_date'].astype(
        str) + '_' + stats_23_24['match_key']

    # *** Drop duplicate matchups
    stats_23_24_new = stats_23_24.drop_duplicates(
        subset='game_id').reset_index(drop=True)

    # *** Rename over/under columns
    stats_23_24_new.rename(
        columns={'OU': 'ou', 'Total': 'ou_total'}, inplace=True)

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

    # *** Drop unneeded columns
    stats_23_24_new.drop(['Team', 'Opponent'], inplace=True, axis=1)

    # Determine who the favorite/underdog is based on
    # spread value (+/-) and home/away
    stats_23_24_new[['fav_side', 'dog_side']] = stats_23_24_new.apply(
        get_roles, axis=1)

    # *** Calculate absolute value of 'spread' value and if the favorite
    # covered the spread or not
    stats_23_24_new['spread'] = stats_23_24_new['spread'].abs()
    stats_23_24_new['favorite_covered'] = stats_23_24_new.apply(
        favorite_covered, axis=1)

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

    stats_23_24_new.drop(columns=[
        'Game', 'Result', 'Score', 'ATS'], axis=1, inplace=True)

    stats_23_24_new['ml_fav'] = pd.Series(dtype='Int64')
    stats_23_24_new['ml_dog'] = pd.Series(dtype='Int64')

    # Clean team names, putting spaces in between words of school name
    # correcting Miami Florida and Miami Ohio to Miami FL and Miami OH to
    # make it consistent across the other datasets.  Change 'St' to 'State'
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

    # Apply mark_bad_pair function
    mark_bad_pair(hist_odds)

    # Combine both rows in a contest in the hist_odds DataFrame into one row

    hist_odds['month'] = pd.Series(dtype='Int64')
    hist_odds['day'] = pd.Series(dtype='Int64')

    hist_odds['fav_side'] = pd.Series(dtype='str')
    hist_odds['dog_side'] = pd.Series(dtype='str')

    for i in range(0, len(hist_odds) - 1, 2):
        raw_date_str = str(hist_odds.loc[i, 'Date']).zfill(4)
        hist_odds.loc[i, 'raw_date_str'] = raw_date_str
        hist_odds.loc[i, 'month'] = int(raw_date_str[:-2])
        hist_odds.loc[i, 'day'] = int(raw_date_str[-2:])
        hist_odds.loc[i, 'full_date'] = (str(
            hist_odds.loc[i, 'Season']) + '-' + str(
            hist_odds.loc[i, 'month']) + '-' + str(hist_odds.loc[i, 'day']))

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
        # and combine over/under and spread on to one line depending on which
        # number is higher (the spread number is ALWAYS
        # going to be less than the over/under)
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

    # *** Populate teams in the 'away_team' column that are not FBS with the
    # 'FCS' designation
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
    hist_odds.reset_index(drop=True, inplace=True)

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

    # *** Determine the money line favorite and underdog
    hist_odds['ml_fav'] = pd.Series(dtype='Int64')
    hist_odds['ml_dog'] = pd.Series(dtype='Int64')

    for i in range(0, len(hist_odds) - 1, 2):
        if (hist_odds.loc[i, 'ML'] != 'NL') & (
                        hist_odds.loc[i+1, 'ML'] != 'NL'):
            if hist_odds.loc[i, 'ML'] < 0:
                hist_odds.loc[i, 'ml_fav'] = hist_odds.loc[i, 'ML']
            else:
                hist_odds.loc[i, 'ml_dog'] = hist_odds.loc[i, 'ML']

            if hist_odds.loc[i+1, 'ML'] < 0:
                hist_odds.loc[i, 'ml_fav'] = hist_odds.loc[i+1, 'ML']
            else:
                hist_odds.loc[i, 'ml_dog'] = hist_odds.loc[i+1, 'ML']

    hist_odds.dropna(subset=['home', 'away'], axis=0).reset_index(
        drop=True, inplace=True)
    hist_odds.drop(
        ['Rot', 'VH', 'Team', 'Final', 'ML', 'Close'],
        axis=1, errors='ignore', inplace=True).reset_index(drop=True)

    # *** Create match key from teams in contest
    hist_odds['match_key'] = hist_odds.apply(make_match_key, axis=1)
    hist_odds['game_id'] = hist_odds['full_date'].astype(
        str) + '_' + hist_odds['match_key']

    # *** Drop duplicate matchups
    hist_odds.drop_duplicates(
        subset='game_id', inplace=True).reset_index(drop=True)

    hist_odds['full_date'] = pd.to_datetime(
        hist_odds['full_date'], errors="coerce")

    hist_odds.columns = [col.lower().strip() for col in hist_odds.columns]

    # *** Determine if the favorite covered the spread
    hist_odds['favorite_covered'] = hist_odds.apply(favorite_covered, axis=1)

    # *** Drop unneedefd columns from historical odds dataset
    hist_odds.drop(['season', 'date', 'month', 'day', 'raw_date_str'], axis=1,
                   inplace=True, errors='ignore')

    def build_spread_means(valid_rows):
        spread_means = valid_rows.groupby('spread').agg({
            'ml_fav': 'mean',
            'ml_dog': 'mean'
        }).rename(columns={
            'ml_fav': 'mean_fav_ml',
            'ml_dog': 'mean_dog_ml'
        })

        spread_means['spread'] = spread_means.index
        spread_means.dropna(inplace=True, axis=0)

        spread_means.loc[0.0, ['mean_fav_ml']] = -110
        spread_means.loc[0.0, ['mean_dog_ml']] = -110

        return spread_means

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

    # *** Calculate mean of existing money line favorites and underdogs by
    # spread to impute in missing money line fields
    # Only include rows where both ml_fav and ml_dog exist
    valid_rows = hist_odds[hist_odds['ml_fav'].notna() & hist_odds[
        'ml_dog'].notna()]
    spread_means = build_spread_means(valid_rows)
    ml_conv = spread_means.copy().reset_index(drop=True)

    stats_23_24_new = stats_23_24_new.apply(fill_all_mls, axis=1)

    hist_odds.apply(fill_missing_mls, axis=1, inplace=True)

    stats_23_24_new.dropna(subset=['ml_fav', 'ml_dog'], inplace=True)

    hist_odds.dropna(subset=['ml_fav', 'ml_dog'], inplace=True)

    ml_conv.rename(
        columns={'mean_fav_ml': 'ml_fav', 'mean_dog_ml': 'ml_dog'},
        inplace=True)

    stats_23_24_new['spread'] = stats_23_24_new['spread'].abs()

    # *** Ensure all columns in the 'ml_conv' DataFrame are numeric and that
    # 'fav_ml' and 'dog_ml' columns are integers
    ml_conv['spread'] = pd.to_numeric(ml_conv['spread'])
    ml_conv['ml_fav'] = pd.to_numeric(ml_conv['ml_fav']).round()
    ml_conv['ml_dog'] = pd.to_numeric(ml_conv['ml_dog']).round()

    ml_conv.dropna().reset_index(drop=True)

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
    box_scores['full_date'] = pd.to_datetime(box_scores['full_date'],
                                             errors="coerce")

    # *** Create match_key column consisting of both teams
    box_scores['match_key'] = box_scores.apply(make_match_key, axis=1)
    box_scores['game_id'] = box_scores['full_date'].astype(
        str) + '_' + box_scores['match_key']

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
    box_scores.reset_index(drop=True, inplace=True)

    # *** Impute 'n' for games where they are not rivalries
    for i in range(0, len(box_scores)):
        if pd.isna(box_scores.loc[i, 'rivalry']):
            box_scores.loc[i, 'rivalry'] = 'n'

    stad_capacity.drop('Stadium', axis=1, inplace=True)
    stad_capacity.set_index('home', inplace=True)

    # *** Impute capacity from stad_capacity into box_scores
    for i, row in box_scores.iterrows():
        home_team = row['home']
        capacity = stad_capacity.loc[home_team, 'capacity']
        if pd.isna(box_scores.loc[i, 'capacity']):
            box_scores.at[i, 'capacity'] = capacity

    # Convert to 'capacity' to numeric
    box_scores['capacity'] = pd.to_numeric(
        box_scores['capacity'], errors='coerce').astype('Int64')

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
        box_scores['attend_percent'] = attend_percent

    stats_23_24_new.reset_index(inplace=True)

    hist_odds.reset_index(inplace=True)

    box_scores.reset_index(inplace=True)

    stats_23_24_new.sort_values('game_id', axis=0, inplace=True)

    hist_odds.sort_values('game_id', axis=0, inplace=True)

    box_scores.sort_values('game_id', axis=0, inplace=True)

    cols_to_fix = ['home_score', 'away_score']

    hist_odds[cols_to_fix] = hist_odds[cols_to_fix].astype('Int64')
    stats_23_24_new[cols_to_fix] = stats_23_24_new[cols_to_fix].astype('Int64')

    # Merge stats_23_24 and hist_ods DataFrames
    merged_odds_df = pd.concat([hist_odds, stats_23_24_new])

    # *** Create match_key column consisting of both teams
    merged_odds_df['match_key'] = merged_odds_df.apply(make_match_key, axis=1)
    merged_odds_df['game_id'] = merged_odds_df['full_date'].astype(
        str) + '_' + merged_odds_df['match_key']

    # Merge odds and box scores DataFrames
    merged_df = box_scores.merge(merged_odds_df, how='inner', on='game_id')

    merged_df.dropna(inplace=True)

    merged_df.drop(['full_date_y', 'index_y', 'index_x', 'match_key_y', 'away_y', 'home_y', 'match_key', 'away_score', 'home_score'], inplace=True, axis=1, errors='ignore')

    merged_df.rename(columns={'match_key_x': 'match_key', 'full_date_x': 'full_date', 'home_x': 'home', 'away_x': 'away'}, inplace=True)

    return merged_df


if __name__ == "__main__":

    go()
