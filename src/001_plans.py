
import gc
import pandas as pd
import numpy as np
import warnings

from tqdm import tqdm
from utils import loadJSON, FlattenDataSimple, save2pkl

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
    plans_df = [FlattenDataSimple(plans, key) for key in tqdm(['distance', 'price', 'eta', 'transport_mode'])]
    plans_df = pd.concat(plans_df,axis=1)

    # merge plan_time & click_mode
    plans_df = pd.merge(plans_df.reset_index(), plans[['sid','plan_time', 'click_mode']], on='sid',how='outer')

    # cleaning
    for c in plans_df.columns.to_list():
        if 'price' in c:
            plans_df[c] = plans_df[c].replace('',0)

    plans_df['plan_time'] = pd.to_datetime(plans_df['plan_time'])

    # datetime features
    plans_df['plan_weekday'] = plans_df['plan_time'].dt.weekday
    plans_df['plan_hour'] = plans_df['plan_time'].dt.hour
    plans_df['plan_weekday_count'] = plans_df['plan_weekday'].map(plans_df['plan_weekday'].value_counts())
    plans_df['plan_hour_count'] = plans_df['plan_hour'].map(plans_df['plan_hour'].value_counts())

    # target encoding
    train_plans = plans_df[plans_df['click_mode'].notnull()]
    target_dummies=pd.get_dummies(train_plans.click_mode.astype(int), prefix='target')
    cols_target = ['plans_{}_transport_mode'.format(i) for i in range(0,7)]
    cols_dummies = target_dummies.columns.to_list()
    train_plans = pd.concat([train_plans, target_dummies],axis=1)
    for i,d in tqdm(enumerate(cols_dummies)):
        df_g = train_plans[cols_target+cols_dummies].groupby(d).mean()
        for c in cols_target:
            plans_df['{}_target_{}'.format(c,i)]=plans_df[c].map(df_g[c])

    # save as pkl
    save2pkl('../features/plans.pkl', plans_df)

if __name__ == '__main__':
    main()
