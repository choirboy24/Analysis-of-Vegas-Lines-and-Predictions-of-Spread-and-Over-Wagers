# Analysis-of-Vegas-Lines-and-Predictions-of-Spread-and-Over-Wagers

## Project Overview

This project involved creating prediction models for Vegas lines for college football (spread, over/under, and moneylines).  I made use of Python, Jupyter Notebooks, in addition to Pandas, NumPy, as well as various other Python libraries.

---

## Contributing

I welcome any feedback or contributions.  I've always said you don't learn anything by hearing yourself speak (or type).  If you'd like to suggest a major change or offer feedback, please open an issue first so we can discuss it.

---

## License

This project and its data is released to the public domain.
Feel free to use, remix, or build on it.

---

## Project Status

Originally completed as part of coursework at Western Governors University, this project is now open for further exploration.

Future enhancements may include:
-a UI built using Tkinter or PyQt5
-additional historical data to refine the prediction models, such as key injuries, win/loss record between the teams, computer rankings (Massey, Sagarin, etc.)
-dashboards

---

## Raw Data
2023_2024_lines.txt - this dataset has the over/under and spread lines for each FBS game in 2023 and 2024.  To be combined with merged_cfb_historical_odds.csv

merged_cfb_historical_odds.csv - this dataset has the over/under and spread lines for each FBS game from 2007 to 2022.  To be combined with 2023_2024_lines.txt

cfb_box_scores_2007_2024.csv - this dataset has the box scores and every key statistic for each FBS game from 2007 to 2024.  To be combined with 2023_2024_lines.txt and merged_cfb_historical_odds.csv after cleaning/imputation

stadium_capacity.csv - this dataset has the capacity of each FBS stadium as of 2024

ml_conv.csv - this dataset will get populated with an average of both the favorite moneyline and underdog moneyline for each spread from 1.0 up to the highest spread line available in the hist_odds dataset

---

## Cleaned Data

merged_df.csv - contains the cleaned and combined dataset
