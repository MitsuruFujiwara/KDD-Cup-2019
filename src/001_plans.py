
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
    train_plans = pd.merge(train_plans, train_clicks[['sid','click_mode']], on='sid', how='left')

    # fill na (no click)
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
    plans_df['plan_weekday_hour'] = plans_df['plan_weekday'].astype(str)+'_'+plans_df['plan_hour'].astype(str)
    plans_df['plan_time_diff'] = plans_df.index.map(plans_df.sort_values('plan_time')['plan_time'].diff().dt.seconds)

    # factorize
    plans_df['plan_weekday_hour'], _ = pd.factorize(plans_df['plan_weekday_hour'])

    plans_df['plan_weekday_count'] = plans_df['plan_weekday'].map(plans_df['plan_weekday'].value_counts())
    plans_df['plan_hour_count'] = plans_df['plan_hour'].map(plans_df['plan_hour'].value_counts())
    plans_df['plan_weekday_hour_count'] = plans_df['plan_weekday_hour'].map(plans_df['plan_weekday_hour'].value_counts())

    # stats features
    cols_distance = ['plan_{}_distance'.format(i) for i in range(0,7)]
    cols_price = ['plan_{}_price'.format(i) for i in range(0,7)]
    cols_eta = ['plan_{}_eta'.format(i) for i in range(0,7)]

    plans_df['plan_distance_mean'] = plans_df[cols_distance].mean(axis=1)
    plans_df['plan_distance_sum'] = plans_df[cols_distance].sum(axis=1)
    plans_df['plan_distance_max'] = plans_df[cols_distance].max(axis=1)
    plans_df['plan_distance_min'] = plans_df[cols_distance].min(axis=1)
    plans_df['plan_distance_var'] = plans_df[cols_distance].var(axis=1)
    plans_df['plan_distance_skew'] = plans_df[cols_distance].skew(axis=1)

    plans_df['plan_price_mean'] = plans_df[cols_price].mean(axis=1)
    plans_df['plan_price_sum'] = plans_df[cols_price].sum(axis=1)
    plans_df['plan_price_max'] = plans_df[cols_price].max(axis=1)
    plans_df['plan_price_min'] = plans_df[cols_price].min(axis=1)
    plans_df['plan_price_var'] = plans_df[cols_price].var(axis=1)
    plans_df['plan_price_skew'] = plans_df[cols_price].skew(axis=1)

    plans_df['plan_eta_mean'] = plans_df[cols_eta].mean(axis=1)
    plans_df['plan_eta_sum'] = plans_df[cols_eta].sum(axis=1)
    plans_df['plan_eta_max'] = plans_df[cols_eta].max(axis=1)
    plans_df['plan_eta_min'] = plans_df[cols_eta].min(axis=1)
    plans_df['plan_eta_var'] = plans_df[cols_eta].var(axis=1)
    plans_df['plan_eta_skew'] = plans_df[cols_eta].skew(axis=1)

    # min-max plan (categorical)
    plans_df['plan_distance_max_plan'] = plans_df[cols_distance].idxmax(axis=1)
    plans_df['plan_distance_min_plan'] = plans_df[cols_distance].idxmin(axis=1)
    plans_df['plan_price_max_plan'] = plans_df[cols_price].idxmax(axis=1)
    plans_df['plan_price_min_plan'] = plans_df[cols_price].idxmin(axis=1)
    plans_df['plan_eta_max_plan'] = plans_df[cols_eta].idxmax(axis=1)
    plans_df['plan_eta_min_plan'] = plans_df[cols_eta].idxmin(axis=1)

    # factorize
    plans_df['plan_distance_max_plan'], _ = pd.factorize(plans_df['plan_distance_max_plan'])
    plans_df['plan_distance_min_plan'], _ = pd.factorize(plans_df['plan_distance_min_plan'])
    plans_df['plan_price_max_plan'], _ = pd.factorize(plans_df['plan_price_max_plan'])
    plans_df['plan_price_min_plan'], _ = pd.factorize(plans_df['plan_price_min_plan'])
    plans_df['plan_eta_max_plan'], _ = pd.factorize(plans_df['plan_eta_max_plan'])
    plans_df['plan_eta_min_plan'], _ = pd.factorize(plans_df['plan_eta_min_plan'])

    # count features
    plans_df['plan_distance_max_plan_count'] = plans_df['plan_distance_max_plan'].map(plans_df['plan_distance_max_plan'].value_counts())
    plans_df['plan_distance_min_plan_count'] = plans_df['plan_distance_min_plan'].map(plans_df['plan_distance_min_plan'].value_counts())
    plans_df['plan_price_max_plan_count'] = plans_df['plan_price_max_plan'].map(plans_df['plan_price_max_plan'].value_counts())
    plans_df['plan_price_min_plan_count'] = plans_df['plan_price_min_plan'].map(plans_df['plan_price_min_plan'].value_counts())
    plans_df['plan_eta_max_plan_count'] = plans_df['plan_eta_max_plan'].map(plans_df['plan_eta_max_plan'].value_counts())
    plans_df['plan_eta_min_plan_count'] = plans_df['plan_eta_min_plan'].map(plans_df['plan_eta_min_plan'].value_counts())

    # count features
    cols_mode = ['plan_{}_transport_mode'.format(i) for i in range(0,7)]
    for c in cols_mode:
        plans_df[c+'_count'] = plans_df[c].map(plans_df[c].value_counts())

    # number features
    plans_df['plan_num_plans'] = plans_df[cols_mode].notnull().sum(axis=1)
    plans_df['plan_num_free_plans'] = (plans_df[cols_price]==0).sum(axis=1)

    # ratio features
    for i in range(0,7):
        plans_df['plan_{}_price_distance_ratio'.format(i)] = plans_df['plan_{}_price'.format(i)] / plans_df['plan_{}_distance'.format(i)]
        plans_df['plan_{}_price_eta_ratio'.format(i)] = plans_df['plan_{}_price'.format(i)] / plans_df['plan_{}_eta'.format(i)]
        plans_df['plan_{}_distance_eta_ratio'.format(i)] = plans_df['plan_{}_distance'.format(i)] / plans_df['plan_{}_eta'.format(i)]

    # stats features of ratio
    cols_price_distance_ratio = ['plan_{}_price_distance_ratio'.format(i) for i in range(0,7)]
    cols_price_eta_ratio = ['plan_{}_price_eta_ratio'.format(i) for i in range(0,7)]
    cols_distance_eta_ratio = ['plan_{}_distance_eta_ratio'.format(i) for i in range(0,7)]

    plans_df['plan_price_distance_ratio_mean'] = plans_df[cols_price_distance_ratio].mean(axis=1)
    plans_df['plan_price_distance_ratio_sum'] = plans_df[cols_price_distance_ratio].sum(axis=1)
    plans_df['plan_price_distance_ratio_max'] = plans_df[cols_price_distance_ratio].max(axis=1)
    plans_df['plan_price_distance_ratio_min'] = plans_df[cols_price_distance_ratio].min(axis=1)
    plans_df['plan_price_distance_ratio_var'] = plans_df[cols_price_distance_ratio].var(axis=1)
    plans_df['plan_price_distance_ratio_skew'] = plans_df[cols_price_distance_ratio].skew(axis=1)

    plans_df['plan_price_eta_ratio_mean'] = plans_df[cols_price_eta_ratio].mean(axis=1)
    plans_df['plan_price_eta_ratio_sum'] = plans_df[cols_price_eta_ratio].sum(axis=1)
    plans_df['plan_price_eta_ratio_max'] = plans_df[cols_price_eta_ratio].max(axis=1)
    plans_df['plan_price_eta_ratio_min'] = plans_df[cols_price_eta_ratio].min(axis=1)
    plans_df['plan_price_eta_ratio_var'] = plans_df[cols_price_eta_ratio].var(axis=1)
    plans_df['plan_price_eta_ratio_skew'] = plans_df[cols_price_eta_ratio].skew(axis=1)

    plans_df['plan_distance_eta_ratio_mean'] = plans_df[cols_distance_eta_ratio].mean(axis=1)
    plans_df['plan_distance_eta_ratio_sum'] = plans_df[cols_distance_eta_ratio].sum(axis=1)
    plans_df['plan_distance_eta_ratio_max'] = plans_df[cols_distance_eta_ratio].max(axis=1)
    plans_df['plan_distance_eta_ratio_min'] = plans_df[cols_distance_eta_ratio].min(axis=1)
    plans_df['plan_distance_eta_ratio_var'] = plans_df[cols_distance_eta_ratio].var(axis=1)
    plans_df['plan_distance_eta_ratio_skew'] = plans_df[cols_distance_eta_ratio].skew(axis=1)

    # min-max plan (categorical) for ratio features
    plans_df['plan_price_distance_ratio_max_plan'] = plans_df[cols_price_distance_ratio].idxmax(axis=1)
    plans_df['plan_price_distance_ratio_min_plan'] = plans_df[cols_price_distance_ratio].idxmin(axis=1)
    plans_df['plan_price_eta_ratio_max_plan'] = plans_df[cols_price_eta_ratio].idxmax(axis=1)
    plans_df['plan_price_eta_ratio_min_plan'] = plans_df[cols_price_eta_ratio].idxmin(axis=1)
    plans_df['plan_distance_eta_ratio_max_plan'] = plans_df[cols_distance_eta_ratio].idxmax(axis=1)
    plans_df['plan_distance_eta_ratio_min_plan'] = plans_df[cols_distance_eta_ratio].idxmin(axis=1)

    # factorize
    plans_df['plan_price_distance_ratio_max_plan'], _ = pd.factorize(plans_df['plan_price_distance_ratio_max_plan'])
    plans_df['plan_price_distance_ratio_min_plan'], _ = pd.factorize(plans_df['plan_price_distance_ratio_min_plan'])
    plans_df['plan_price_eta_ratio_max_plan'], _ = pd.factorize(plans_df['plan_price_eta_ratio_max_plan'])
    plans_df['plan_price_eta_ratio_min_plan'], _ = pd.factorize(plans_df['plan_price_eta_ratio_min_plan'])
    plans_df['plan_distance_eta_ratio_max_plan'], _ = pd.factorize(plans_df['plan_distance_eta_ratio_max_plan'])
    plans_df['plan_distance_eta_ratio_min_plan'], _ = pd.factorize(plans_df['plan_distance_eta_ratio_min_plan'])

    # count features
    plans_df['plan_price_distance_ratio_max_plan_count'] = plans_df['plan_price_distance_ratio_max_plan'].map(plans_df['plan_price_distance_ratio_max_plan'].value_counts())
    plans_df['plan_price_distance_ratio_min_plan_count'] = plans_df['plan_price_distance_ratio_min_plan'].map(plans_df['plan_price_distance_ratio_min_plan'].value_counts())
    plans_df['plan_price_eta_ratio_max_plan_count'] = plans_df['plan_price_eta_ratio_max_plan'].map(plans_df['plan_price_eta_ratio_max_plan'].value_counts())
    plans_df['plan_price_eta_ratio_min_plan_count'] = plans_df['plan_price_eta_ratio_min_plan'].map(plans_df['plan_price_eta_ratio_min_plan'].value_counts())
    plans_df['plan_distance_eta_ratio_max_plan_count'] = plans_df['plan_distance_eta_ratio_max_plan'].map(plans_df['plan_distance_eta_ratio_max_plan'].value_counts())
    plans_df['plan_distance_eta_ratio_min_plan_count'] = plans_df['plan_distance_eta_ratio_min_plan'].map(plans_df['plan_distance_eta_ratio_min_plan'].value_counts())

    # target encoding
    cols_transport_mode = ['plan_{}_transport_mode'.format(i) for i in range(0,5)]

    train_plans = plans_df[plans_df['click_mode'].notnull()]
    target_dummies=pd.get_dummies(train_plans.click_mode.astype(int), prefix='target')
    cols_dummies = target_dummies.columns.to_list()
    train_plans = pd.concat([train_plans, target_dummies],axis=1)
    cols_target_encoding = ['plan_weekday','plan_hour','plan_weekday_hour', 'plan_num_plans', 'plan_num_free_plans']
    for c in tqdm(cols_target_encoding+cols_transport_mode):
        df_g = train_plans[[c]+cols_dummies].groupby(c).mean()
        for i,d in enumerate(cols_dummies):
            plans_df['{}_target_{}'.format(c,i)]=plans_df[c].map(df_g[d])

    # save as pkl
    save2pkl('../features/plans.pkl', plans_df)

if __name__ == '__main__':
    main()
