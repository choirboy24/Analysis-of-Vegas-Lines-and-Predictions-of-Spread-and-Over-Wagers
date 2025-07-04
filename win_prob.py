import numpy as np
from functions import spread_to_moneyline, spread_to_win_prob, implied_prob, moneyline_to_prob

final_prob = 0.7 * model_prediction + 0.3 * spread_based_prob

# Apply to both teams
stats_23_24['home_implied_prob'] = stats_23_24['home_ml'].apply(implied_prob)
stats_23_24['away_implied_prob'] = stats_23_24['away_ml'].apply(implied_prob)