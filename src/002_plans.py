
import gc
import pandas as pd
import numpy as np
import warnings

from tqdm import tqdm
from utils import loadJSON

warnings.filterwarnings('ignore')

#==============================================================================
# Preprocessing Plans
#==============================================================================

def main(num_rows=None):
    # load csv
    train_plans = pd.read_csv('../input/data_set_phase1/train_plans.csv',nrows=num_rows)
    test_plans = pd.read_csv('../input/data_set_phase1/test_plans.csv',nrows=num_rows)
    train_clicks = pd.read_csv('../input/data_set_phase1/train_clicks.csv')

    # merge click
    train_plans = pd.merge(train_plans, train_clicks[['sid','click_mode']], on='sid', how='outer')

    # fill na
    train_plans['click_mode'].fillna(0, inplace=True)

    # set test target as nan
    test_plans['click_mode'] = np.nan

    # merge train & test
    plans = train_plans.append(test_plans)

    del train_plans, test_plans
    gc.collect()

    # reset index
    plans.reset_index(inplace=True,drop=True)

    # load JSON
    for key in tqdm(['distance', 'price', 'eta', 'transport_mode']):
        plans[key] = plans.plans.apply(lambda x: loadJSON(x,key))

    # TODO: Preprocessing

    # save as pkl
#    save2pkl('../features/plans.pkl', plans_df)
    print(plans)

if __name__ == '__main__':
    main()
