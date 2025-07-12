# Not for version 1.0.0...will be for version 1.1.0

# *** Import required libraries

import os
import pandas as pd
import re


project_path = os.getcwd()
data_path_strength_sched = os.path.join(
    project_path,
    'data/strength_of_schedule.txt'
)

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
strength_schedule.rename(columns={'Date': 'full_date', 'Team': 'team',
                                  'Rating': 'rating', 'Season': 'season'},
                         inplace=True)

# *** Extract wins and losses for each category of strength of schedule
#  and put them in their own columns
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


strength_schedule['v_1_10_wins'] = pd.Series(v_1_10_wins)
strength_schedule['v_1_10_losses'] = pd.Series(v_1_10_losses)
strength_schedule['v_11_25_wins'] = pd.Series(v_11_25_wins)
strength_schedule['v_11_25_losses'] = pd.Series(v_11_25_losses)
strength_schedule['v_26_40_wins'] = pd.Series(v_26_40_wins)
strength_schedule['v_26_40_losses'] = pd.Series(v_26_40_losses)

strength_schedule.drop(['v 1-10', 'v 11-25', 'v 26-40'], inplace=True, axis=1)
