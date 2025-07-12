## Model Details

Model date: 07/17/2025
Model version: 1.0.0
Model type: Machine learning model using ### RandomForestClassifier as the training model

## Intended Use

The model is used to be able to predict the likelihood either the favorite or the underdog of a college football contest of covering the spread and whether the over or the under of the contest will be met.

## Training Data

80% of the data from merged_df.csv compiled from various websites that contain historical Vegas betting lines and box scores

## Evaluation Data

20% of the data from merged_df.csv compiled from various websites that contain historical Vegas betting lines and box scores

## Metrics

Spread: Accuracy:   | F1: 
Over/Under: Accuracy  | F1:
Underdog Wins: Accuracy  | F1: 

## Ethical Considerations

Even though this section is labeled 'ethical considerations', for this model, this section could really be relabeled 'moral considerations'.  This is because different segments of the population, whether it be from a religious standpoint, financial limitations, or tendency towards addictive behaviors, may find sports betting reprehensible.  These segments would most likely not be making use of this model since they most likely would not be placing any wagers (or they live in a state where sports betting is not legal).  Even if someone wouldn't be placing actual wagers, they could, in theory, use it to make hypothetical bets to see if their guess would be correct based on the results of the model.

## Caveats and Recommendations

During the college football season, dozens of new games will be added to the base dataset every week, with its box score statistics and the original moneyline, spread, and over/under lines.  Because of this, the model should be retrained every week during the college football season until the national championship game is completed.