import numpy as np
import pandas as pd


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


def spread_to_win_prob(spread):
    # Positive spread = underdog
    # Negative spread = favorite
    return 1 / (1 + np.exp(-0.31 * spread))


def moneyline_to_prob(moneyline):
    if moneyline < 0:
        return -moneyline / (-moneyline + 100)
    else:
        return 100 / (moneyline + 100)


def spread_to_moneyline(spread):
    # spread is negative for favorites
    win_prob = 1 / (1 + np.exp(-0.31 * spread))
    if win_prob >= 0.5:
        return -100 * win_prob / (1 - win_prob)
    else:
        return 100 * (1 - win_prob) / win_prob


def implied_prob(ml):
    if pd.isna(ml):
        return None
    if ml < 0:
        return round((-ml) / (-ml + 100), 4)
    else:
        return round(100 / (ml + 100), 4)


def make_match_key(row):
    teams = [str(row['home']), str(row['away'])]
    teams = [team.strip() for team in teams if team and team != 'nan']
    return '_'.join(sorted(teams))


def test_match_key(row):
    try:
        _ = '_'.join(sorted([row['home'], row['away']]))
        return False  # No error
    except Exception as e:
        print(f"Error on row:\n{row}\nException: {e}")
        return True

