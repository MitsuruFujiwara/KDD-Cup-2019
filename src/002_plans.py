
import gc
import pandas as pd
import numpy as np
import warnings

from tqdm import tqdm
from utils import loadJSON, FlattenData, save2pkl

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

    # convert json
    for key in tqdm(['distance', 'price', 'eta', 'transport_mode']):
        plans[key] = plans.plans.apply(lambda x: loadJSON(x,key))

    # flatten
    plans_df = [FlattenData(plans, key) for key in tqdm(['distance', 'price', 'eta', 'transport_mode'])]
    plans_df = pd.concat(plans_df,axis=1)

    # drop na
    plans_df.dropna(inplace=True)

    # merge plan_time & click_mode
    plans_df = pd.merge(plans_df.reset_index(), plans[['sid','plan_time', 'click_mode']], on='sid',how='outer')

    # set target
    plans_df['target'] = (plans_df['transport_mode']==plans_df['click_mode']).astype(int)

    # cleaning
    plans_df['price'] = plans_df['price'].replace('',0)
    plans_df['plan_time'] = pd.to_datetime(plans_df['plan_time'])

    # datetime features
    plans_df['weekday'] = plans_df['plan_time'].dt.weekday
    plans_df['hour'] = plans_df['plan_time'].dt.hour
    plans_df['weekday_count'] = plans_df['weekday'].map(plans_df['weekday'].value_counts())
    plans_df['hour_count'] = plans_df['hour'].map(plans_df['hour'].value_counts())

    # features
    # TODO:

    # save as pkl
    save2pkl('../features/plans.pkl', plans_df)

if __name__ == '__main__':
    main()
