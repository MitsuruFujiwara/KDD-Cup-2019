
import gc
import pandas as pd
import numpy as np
import warnings

from utils import save2pkl

warnings.filterwarnings('ignore')

#==============================================================================
# Preprocessing Queries
#==============================================================================

def main(num_rows=None):
    # load csv
    profiles = pd.read_csv('../input/data_set_phase1/profiles.csv')

    # feature engineering
    feats = [f for f in profiles.columns.to_list() if f not in ['pid']]

    profiles['p_sum'] = profiles[feats].mean(axis=1)
    profiles['p_mean'] = profiles[feats].sum(axis=1)
    profiles['p_std'] = profiles[feats].std(axis=1)

    profiles['p_sum_count'] = profiles['p_sum'].map(profiles['p_sum'].value_counts())

    # save as pkl
    save2pkl('../features/profiles.pkl', profiles)

if __name__ == '__main__':
    main()
