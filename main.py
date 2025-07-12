from eda import go as eda_go
from clean_data import go as clean_data_go, stats_23_24_new, hist_odds, ml_conv, merged_odds_df, merged_df
from feat_engineering import go as feat_engineering_go
from ml.predict import go as predict_go
from ml.model import go as model_go

if __name__ == "__main__":
    # Load the dataset

    # Exploratory Data Analysis
    stats_23_24, hist_odds, box_scores, stad_capacity = eda_go()

    # Data Cleaning and Imputation
    merged_df, ml_conv, box_scores, stats_23_24_new = clean_data_go()

    # Feature Engineering
    feat_engineering_go()

    # Model Creation
    model_go()

    # Prediction
    predict_go()

    # Write separate and merged DataFrames to new CSV files
    stats_23_24_new.to_csv('data/stats_23_24_new.csv')

    hist_odds.to_csv('data/hist_odds.csv')

    ml_conv.to_csv('data/ml_conv.csv')

    merged_odds_df.to_csv('data/merged_odds_df.csv')

    merged_df.to_csv('data/merged_df.csv')