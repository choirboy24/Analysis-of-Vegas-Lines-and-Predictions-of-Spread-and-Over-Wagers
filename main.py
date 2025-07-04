import os
import pandas as pd
import re
from sklearn.impute import SimpleImputer
'''
import math
import numpy as np
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import OneHotEncoder
'''

# Define locations of raw data
project_path = os.path.dirname(os.path.abspath(__file__))
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
data_path_ml_conv = os.path.join(
    project_path,
    'data/ml_conv.csv'
)

# Read in strength_of_schedule.txt to Pandas DataFrame
strength_schedule = pd.read_csv(
    data_path_strength_sched, sep='\t', encoding='ansi')

# Read in lines from 2023 and 2024 seasons into Pandas DataFrame
stats_23_24 = pd.read_csv(data_path_2023_2024_lines, sep='\t')

# Take each contest and look for the corresponding game with
# same teams on same date and delete the second game in stats_23_24

hist_odds = pd.read_csv(data_path_merged_cfb_historical_odds)
hist_odds['Season'] = pd.to_numeric(
    hist_odds['Season'], errors='coerce').fillna(0).astype(int)
hist_odds = hist_odds.drop(
    ['1st', '2nd', '3rd', '4th', 'Open', 'ML', '2H'], axis=1, errors='ignore')


box_scores = pd.read_csv(data_path_cfb_box_scores)
print(box_scores.info())
box_scores = box_scores.drop(['rank_away', 'rank_home', 'q1_away', 'q2_away',
                              'q3_away', 'q4_away', 'ot_away', 'q1_home',
                              'q2_home', 'q3_home', 'q4_home', 'ot_home',
                              'tv'], errors='ignore')

stad_capacity = pd.read_csv(data_path_stad_capacity)
stad_capacity = stad_capacity.drop('Stadium', axis=1)

# Populate list with each unique team in 'Team' column
fbs_teams_list = stats_23_24['Team'].unique().tolist()

'''
for i in box_scores['home']:
    home_team = stad_capacity.loc[stad_capacity['Home Team'] == i]
    capacity =
    box_scores['capacity'] = capacity
'''
home = []

for team in stats_23_24['Opponent']:
    if 'vs ' in team:
        home.append("yes")
        stats_23_24['Opponent'] = stats_23_24['Opponent'].str.replace(
            'vs ', '', regex=False)
    elif '@ ' in team:
        home.append("no")
        stats_23_24['Opponent'] = stats_23_24['Opponent'].str.replace(
            '@ ', '', regex=False)


stats_23_24['Home'] = home
home_scores = []
away_scores = []

for index, row in stats_23_24.iterrows():
    scores = re.findall(r'(\d+)', row['Score'])
    home_score = int(scores[0])
    away_score = int(scores[1])
    if row['Result'] == 'W' and row['Home'] == 'yes':
        home_scores.append(home_score)
        away_scores.append(away_score)
    elif row['Result'] == 'L' and row['Home'] == 'yes':
        home_scores.append(away_score)
        away_scores.append(home_score)
    elif row['Result'] == 'W' and row['Home'] == 'no':
        home_scores.append(away_score)
        away_scores.append(home_score)
    else:
        home_scores.append(home_score)
        away_scores.append(away_score)
stats_23_24['Home Score'] = home_scores
stats_23_24['Away Score'] = away_scores

stats_23_24 = stats_23_24.drop(columns=['Unnamed: 10'])

# Replace FCS (Football Championship Subdivision (formerly Division I-AA))
# opponents with 'FCS'
for index, row in stats_23_24.iterrows():
    if row['Opponent'] not in fbs_teams_list:
        stats_23_24.at[index, 'Opponent'] = 'FCS'
fcs_teams_stats = stats_23_24[stats_23_24['Opponent'] == 'FCS'].index

for index, row in box_scores.iterrows():
    if row['away'] not in fbs_teams_list:
        box_scores.at[index, 'away'] = 'FCS'
fcs_teams_box = box_scores[box_scores['away'] == 'FCS'].index

# Drop games where FCS team is played
stats_23_24.drop(fcs_teams_stats, inplace=True)
stats_23_24 = stats_23_24.reset_index(drop=True)
print(stats_23_24.tail())

box_scores.drop(fcs_teams_box, inplace=True)
box_scores = box_scores.reset_index(drop=True)
print(box_scores.info())

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
