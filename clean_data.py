# *** Import required libraries

import os
import pandas as pd
import re
import math
import numpy as np
import sklearn
from functions import get_roles, get_favorite, favorite_covered, make_match_key, test_match_key


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