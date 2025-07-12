# Import required libraries

import os
import pandas as pd


def go():
    # Set data paths for each input file
    project_path = os.path.dirname(os.path.abspath(__file__))

    # project_path = os.getcwd()
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

    # *** Read in lines from 2023 and 2024 seasons' odds into DataFrame
    stats_23_24 = pd.read_csv(data_path_2023_2024_lines, sep='\t')

    # *** Load historical odds DataFrame
    hist_odds = pd.read_csv(data_path_merged_cfb_historical_odds)

    # Read in cfb_box_scores_2004_2024.csv to DataFrame
    box_scores = pd.read_csv(data_path_cfb_box_scores)

    # *** Load stadium capacity into DataFrame to use for imputation of
    # box_scores['capacity'] column
    stad_capacity = pd.read_csv(data_path_stad_capacity)

    # EDA
    print(stats_23_24.head())
    print(stats_23_24.info())
    print(stats_23_24.describe())

    print(hist_odds.head())
    print(hist_odds.info())
    print(hist_odds.describe())

    print(box_scores.head())
    print(box_scores.info())
    print(box_scores.describe())

    print(stad_capacity.head())
    print(stad_capacity.info())

    return stats_23_24, hist_odds, box_scores, stad_capacity


if __name__ == "__main__":

    go()
