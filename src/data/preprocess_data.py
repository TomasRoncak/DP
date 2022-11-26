import sys
import pandas as pd

sys.path.insert(0, '/Users/tomasroncak/Documents/diplomova_praca/src/')

import constants as const

dataset = pd.read_csv(const.RAW_DATASET_PATH + 'UNSW_NB15_training-set.csv', ignore_index=True)