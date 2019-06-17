
import gc
import pandas as pd
import numpy as np
import sys
import warnings

from chinese_calendar import is_holiday
from tqdm import tqdm

from utils import loadJSON, FlattenDataSimple, to_pickles, line_notify, targetEncodingMultiClass, reduce_mem_usage

warnings.filterwarnings('ignore')

#==============================================================================
# Preprocessing Plans
#==============================================================================

def main(num_rows=None):
    # load csv
    train_plans = pd.read_csv('../input/data_set_phase2/train_plans_phase2.csv',nrows=num_rows)
    test_plans = pd.read_csv('../input/data_set_phase2/test_plans.csv',nrows=num_rows)
    train_clicks = pd.read_csv('../input/data_set_phase2/train_clicks_phase2.csv')

    # phase 1 csv
    train_plans1 = pd.read_csv('../input/data_set_phase2/train_plans_phase1.csv')
    train_clicks1 = pd.read_csv('../input/data_set_phase2/train_clicks_phase1.csv')

    # merge click
    train_plans = pd.merge(train_plans, train_clicks[['sid','click_mode']], on='sid', how='left')
    train_plans1 = pd.merge(train_plans1, train_clicks1[['sid','click_mode']], on='sid', how='left')

    # merge phase 1 data
    train_plans = train_plans1.append(train_plans)

    # fill na (no click)
    train_plans['click_mode'].fillna(0, inplace=True)

    # set test target as nan
    test_plans['click_mode'] = np.nan

    # merge train & test
    plans = train_plans.append(test_plans)

    del train_plans, test_plans, train_plans1, train_clicks, train_clicks1
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

    del plans
    gc.collect()

    # reduce memory usage
    plans_df = reduce_mem_usage(plans_df)

    # cleaning
    for c in plans_df.columns.to_list():
        if 'price' in c:
            plans_df[c] = plans_df[c].replace('',0)

    plans_df['plan_time'] = pd.to_datetime(plans_df['plan_time'])

    # datetime features
    plans_df['plan_weekday'] = plans_df['plan_time'].dt.weekday
    plans_df['plan_hour'] = plans_df['plan_time'].dt.hour
    plans_df['plan_is_holiday'] = plans_df['plan_time'].apply(lambda x: is_holiday(x)).astype(int)
    plans_df['plan_weekday_hour'] = plans_df['plan_weekday'].astype(str)+'_'+plans_df['plan_hour'].astype(str)
    plans_df['plan_is_holiday_hour'] = plans_df['plan_is_holiday'].astype(str)+'_'+plans_df['plan_hour'].astype(str)
    plans_df['plan_time_diff'] = plans_df.index.map(plans_df.sort_values('plan_time')['plan_time'].diff().dt.seconds)

    # factorize
    plans_df['plan_weekday_hour'], _ = pd.factorize(plans_df['plan_weekday_hour'])
    plans_df['plan_is_holiday_hour'], _ = pd.factorize(plans_df['plan_is_holiday_hour'])

    # count features
    plans_df['plan_weekday_count'] = plans_df['plan_weekday'].map(plans_df['plan_weekday'].value_counts())
    plans_df['plan_hour_count'] = plans_df['plan_hour'].map(plans_df['plan_hour'].value_counts())
    plans_df['plan_weekday_hour_count'] = plans_df['plan_weekday_hour'].map(plans_df['plan_weekday_hour'].value_counts())
    plans_df['plan_is_holiday_hour_count'] = plans_df['plan_is_holiday_hour'].map(plans_df['plan_is_holiday_hour'].value_counts())

    # stats features
    cols_transport_mode = ['plan_{}_transport_mode'.format(i) for i in range(0,7)]
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
    plans_df['plan_distance_max_plan'] = plans_df[cols_distance].idxmax(axis=1).apply(lambda x: x[:6]+'_transport_mode' if type(x)==str else np.nan)
    plans_df['plan_distance_min_plan'] = plans_df[cols_distance].idxmin(axis=1).apply(lambda x: x[:6]+'_transport_mode' if type(x)==str else np.nan)
    plans_df['plan_price_max_plan'] = plans_df[cols_price].idxmax(axis=1).apply(lambda x: x[:6]+'_transport_mode' if type(x)==str else np.nan)
    plans_df['plan_price_min_plan'] = plans_df[cols_price].idxmin(axis=1).apply(lambda x: x[:6]+'_transport_mode' if type(x)==str else np.nan)
    plans_df['plan_eta_max_plan'] = plans_df[cols_eta].idxmax(axis=1).apply(lambda x: x[:6]+'_transport_mode' if type(x)==str else np.nan)
    plans_df['plan_eta_min_plan'] = plans_df[cols_eta].idxmin(axis=1).apply(lambda x: x[:6]+'_transport_mode' if type(x)==str else np.nan)

    # map plans
    cols_min_max_plan = ['plan_distance_max_plan','plan_distance_min_plan',
                         'plan_price_max_plan', 'plan_price_min_plan',
                         'plan_eta_max_plan', 'plan_eta_min_plan']
    for c in tqdm(cols_transport_mode):
        for p in cols_min_max_plan:
            plans_df[p][plans_df[p]==c] = plans_df[c][plans_df[p]==c]

    # count features
    plans_df['plan_distance_max_plan_count'] = plans_df['plan_distance_max_plan'].map(plans_df['plan_distance_max_plan'].value_counts())
    plans_df['plan_distance_min_plan_count'] = plans_df['plan_distance_min_plan'].map(plans_df['plan_distance_min_plan'].value_counts())
    plans_df['plan_price_max_plan_count'] = plans_df['plan_price_max_plan'].map(plans_df['plan_price_max_plan'].value_counts())
    plans_df['plan_price_min_plan_count'] = plans_df['plan_price_min_plan'].map(plans_df['plan_price_min_plan'].value_counts())
    plans_df['plan_eta_max_plan_count'] = plans_df['plan_eta_max_plan'].map(plans_df['plan_eta_max_plan'].value_counts())
    plans_df['plan_eta_min_plan_count'] = plans_df['plan_eta_min_plan'].map(plans_df['plan_eta_min_plan'].value_counts())

    # count features
    cols_mode = ['plan_{}_transport_mode'.format(i) for i in range(0,7)]
    cols_mode_count = []
    for c in cols_mode:
        plans_df[c+'_count'] = plans_df[c].map(plans_df[c].value_counts())
        cols_mode_count.append(c+'_count')

    # number features
    plans_df['plan_num_plans'] = plans_df[cols_mode].notnull().sum(axis=1)
    plans_df['plan_num_free_plans'] = (plans_df[cols_price]==0).sum(axis=1)

    # rank features
    plans_df[[ c +'_rank' for c in cols_distance]] = plans_df[cols_distance].rank(axis=1)
    plans_df[[ c +'_rank' for c in cols_price]] = plans_df[cols_price].rank(axis=1)
    plans_df[[ c +'_rank' for c in cols_eta]] = plans_df[cols_eta].rank(axis=1)
    plans_df[[ c +'_rank' for c in cols_mode_count]] = plans_df[cols_mode_count].rank(axis=1)

    # ratio features
    for i in range(0,7):
        plans_df['plan_{}_price_distance_ratio'.format(i)] = plans_df['plan_{}_price'.format(i)] / plans_df['plan_{}_distance'.format(i)]
        plans_df['plan_{}_price_eta_ratio'.format(i)] = plans_df['plan_{}_price'.format(i)] / plans_df['plan_{}_eta'.format(i)]
        plans_df['plan_{}_distance_eta_ratio'.format(i)] = plans_df['plan_{}_distance'.format(i)] / plans_df['plan_{}_eta'.format(i)]

    # prod features
    for i in range(0,7):
        plans_df['plan_{}_price_distance_prod'.format(i)] = plans_df['plan_{}_price'.format(i)] * plans_df['plan_{}_distance'.format(i)]
        plans_df['plan_{}_price_eta_prod'.format(i)] = plans_df['plan_{}_price'.format(i)] * plans_df['plan_{}_eta'.format(i)]
        plans_df['plan_{}_distance_eta_prod'.format(i)] = plans_df['plan_{}_distance'.format(i)] * plans_df['plan_{}_eta'.format(i)]
        plans_df['plan_{}_price_distance_eta_prod'.format(i)] = plans_df['plan_{}_price'.format(i)] * plans_df['plan_{}_distance'.format(i)]* plans_df['plan_{}_eta'.format(i)]

    # ratio features with plan 0
    for i in range(1,7):
        plans_df['plan_{}_distance_ratio_0'.format(i)] = plans_df['plan_{}_distance'.format(i)]/plans_df['plan_0_distance']
        plans_df['plan_{}_price_ratio_0'.format(i)] = plans_df['plan_{}_price'.format(i)]/plans_df['plan_0_price']
        plans_df['plan_{}_eta_ratio_0'.format(i)] = plans_df['plan_{}_eta'.format(i)]/plans_df['plan_0_eta']

        plans_df['plan_{}_price_distance_prod_ratio_0'.format(i)] = plans_df['plan_{}_price_distance_prod'.format(i)] / plans_df['plan_0_price_distance_prod']
        plans_df['plan_{}_price_eta_prod_ratio_0'.format(i)] = plans_df['plan_{}_price_eta_prod'.format(i)] / plans_df['plan_0_price_eta_prod']
        plans_df['plan_{}_distance_eta_prod_ratio_0'.format(i)] = plans_df['plan_{}_distance_eta_prod'.format(i)] / plans_df['plan_0_distance_eta_prod']
        plans_df['plan_{}_price_distance_eta_prod_ratio_0'.format(i)] = plans_df['plan_{}_price_distance_eta_prod'.format(i)] / plans_df['plan_0_price_distance_eta_prod']

    # stats features of ratio
    cols_price_distance_ratio = ['plan_{}_price_distance_ratio'.format(i) for i in range(0,7)]
    cols_price_eta_ratio = ['plan_{}_price_eta_ratio'.format(i) for i in range(0,7)]
    cols_distance_eta_ratio = ['plan_{}_distance_eta_ratio'.format(i) for i in range(0,7)]

    cols_price_distance_prod = ['plan_{}_price_distance_prod'.format(i) for i in range(0,7)]
    cols_price_eta_prod = ['plan_{}_price_eta_prod'.format(i) for i in range(0,7)]
    cols_distance_eta_prod = ['plan_{}_distance_eta_prod'.format(i) for i in range(0,7)]
    cols_price_distance_eta_prod = ['plan_{}_price_distance_eta_prod'.format(i) for i in range(0,7)]

    cols_distance_ratio_0 = ['plan_{}_distance_ratio_0'.format(i) for i in range(1,7)]
    cols_price_ratio_0 = ['plan_{}_price_ratio_0'.format(i) for i in range(1,7)]
    cols_eta_ratio_0 = ['plan_{}_eta_ratio_0'.format(i) for i in range(1,7)]

    cols_price_distance_prod_ratio_0 = ['plan_{}_price_distance_prod_ratio_0'.format(i) for i in range(1,7)]
    cols_price_eta_prod_ratio_0 = ['plan_{}_price_eta_prod_ratio_0'.format(i) for i in range(1,7)]
    cols_distance_eta_prod_ratio_0 = ['plan_{}_distance_eta_prod_ratio_0'.format(i) for i in range(1,7)]
    cols_price_distance_eta_prod_ratio_0 = ['plan_{}_price_distance_eta_prod_ratio_0'.format(i) for i in range(1,7)]

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

    plans_df['plan_price_distance_prod_mean'] = plans_df[cols_price_distance_prod].mean(axis=1)
    plans_df['plan_price_distance_prod_sum'] = plans_df[cols_price_distance_prod].sum(axis=1)
    plans_df['plan_price_distance_prod_max'] = plans_df[cols_price_distance_prod].max(axis=1)
    plans_df['plan_price_distance_prod_min'] = plans_df[cols_price_distance_prod].min(axis=1)
    plans_df['plan_price_distance_prod_var'] = plans_df[cols_price_distance_prod].var(axis=1)
    plans_df['plan_price_distance_prod_skew'] = plans_df[cols_price_distance_prod].skew(axis=1)

    plans_df['plan_price_eta_prod_mean'] = plans_df[cols_price_eta_prod].mean(axis=1)
    plans_df['plan_price_eta_prod_sum'] = plans_df[cols_price_eta_prod].sum(axis=1)
    plans_df['plan_price_eta_prod_max'] = plans_df[cols_price_eta_prod].max(axis=1)
    plans_df['plan_price_eta_prod_min'] = plans_df[cols_price_eta_prod].min(axis=1)
    plans_df['plan_price_eta_prod_var'] = plans_df[cols_price_eta_prod].var(axis=1)
    plans_df['plan_price_eta_prod_skew'] = plans_df[cols_price_eta_prod].skew(axis=1)

    plans_df['plan_distance_eta_prod_mean'] = plans_df[cols_distance_eta_prod].mean(axis=1)
    plans_df['plan_distance_eta_prod_sum'] = plans_df[cols_distance_eta_prod].sum(axis=1)
    plans_df['plan_distance_eta_prod_max'] = plans_df[cols_distance_eta_prod].max(axis=1)
    plans_df['plan_distance_eta_prod_min'] = plans_df[cols_distance_eta_prod].min(axis=1)
    plans_df['plan_distance_eta_prod_var'] = plans_df[cols_distance_eta_prod].var(axis=1)
    plans_df['plan_distance_eta_prod_skew'] = plans_df[cols_distance_eta_prod].skew(axis=1)

    plans_df['plan_price_distance_eta_prod_mean'] = plans_df[cols_price_distance_eta_prod].mean(axis=1)
    plans_df['plan_price_distance_eta_prod_sum'] = plans_df[cols_price_distance_eta_prod].sum(axis=1)
    plans_df['plan_price_distance_eta_prod_max'] = plans_df[cols_price_distance_eta_prod].max(axis=1)
    plans_df['plan_price_distance_eta_prod_min'] = plans_df[cols_price_distance_eta_prod].min(axis=1)
    plans_df['plan_price_distance_eta_prod_var'] = plans_df[cols_price_distance_eta_prod].var(axis=1)
    plans_df['plan_price_distance_eta_prod_skew'] = plans_df[cols_price_distance_eta_prod].skew(axis=1)

    plans_df['plan_distance_ratio_0_mean'] = plans_df[cols_distance_ratio_0].mean(axis=1)
    plans_df['plan_distance_ratio_0_sum'] = plans_df[cols_distance_ratio_0].sum(axis=1)
    plans_df['plan_distance_ratio_0_max'] = plans_df[cols_distance_ratio_0].max(axis=1)
    plans_df['plan_distance_ratio_0_min'] = plans_df[cols_distance_ratio_0].min(axis=1)
    plans_df['plan_distance_ratio_0_var'] = plans_df[cols_distance_ratio_0].var(axis=1)
    plans_df['plan_distance_ratio_0_skew'] = plans_df[cols_distance_ratio_0].skew(axis=1)

    plans_df['plan_price_ratio_0_mean'] = plans_df[cols_price_ratio_0].mean(axis=1)
    plans_df['plan_price_ratio_0_sum'] = plans_df[cols_price_ratio_0].sum(axis=1)
    plans_df['plan_price_ratio_0_max'] = plans_df[cols_price_ratio_0].max(axis=1)
    plans_df['plan_price_ratio_0_min'] = plans_df[cols_price_ratio_0].min(axis=1)
    plans_df['plan_price_ratio_0_var'] = plans_df[cols_price_ratio_0].var(axis=1)
    plans_df['plan_price_ratio_0_skew'] = plans_df[cols_price_ratio_0].skew(axis=1)

    plans_df['plan_eta_ratio_0_mean'] = plans_df[cols_eta_ratio_0].mean(axis=1)
    plans_df['plan_eta_ratio_0_sum'] = plans_df[cols_eta_ratio_0].sum(axis=1)
    plans_df['plan_eta_ratio_0_max'] = plans_df[cols_eta_ratio_0].max(axis=1)
    plans_df['plan_eta_ratio_0_min'] = plans_df[cols_eta_ratio_0].min(axis=1)
    plans_df['plan_eta_ratio_0_var'] = plans_df[cols_eta_ratio_0].var(axis=1)
    plans_df['plan_eta_ratio_0_skew'] = plans_df[cols_eta_ratio_0].skew(axis=1)

    plans_df['plan_price_distance_prod_ratio_0_mean'] = plans_df[cols_price_distance_prod_ratio_0].mean(axis=1)
    plans_df['plan_price_distance_prod_ratio_0_sum'] = plans_df[cols_price_distance_prod_ratio_0].sum(axis=1)
    plans_df['plan_price_distance_prod_ratio_0_max'] = plans_df[cols_price_distance_prod_ratio_0].max(axis=1)
    plans_df['plan_price_distance_prod_ratio_0_min'] = plans_df[cols_price_distance_prod_ratio_0].min(axis=1)
    plans_df['plan_price_distance_prod_ratio_0_var'] = plans_df[cols_price_distance_prod_ratio_0].var(axis=1)
    plans_df['plan_price_distance_prod_ratio_0_skew'] = plans_df[cols_price_distance_prod_ratio_0].skew(axis=1)

    plans_df['plan_price_eta_prod_ratio_0_mean'] = plans_df[cols_price_eta_prod_ratio_0].mean(axis=1)
    plans_df['plan_price_eta_prod_ratio_0_sum'] = plans_df[cols_price_eta_prod_ratio_0].sum(axis=1)
    plans_df['plan_price_eta_prod_ratio_0_max'] = plans_df[cols_price_eta_prod_ratio_0].max(axis=1)
    plans_df['plan_price_eta_prod_ratio_0_min'] = plans_df[cols_price_eta_prod_ratio_0].min(axis=1)
    plans_df['plan_price_eta_prod_ratio_0_var'] = plans_df[cols_price_eta_prod_ratio_0].var(axis=1)
    plans_df['plan_price_eta_prod_ratio_0_skew'] = plans_df[cols_price_eta_prod_ratio_0].skew(axis=1)

    plans_df['plan_distance_eta_prod_ratio_0_mean'] = plans_df[cols_distance_eta_prod_ratio_0].mean(axis=1)
    plans_df['plan_distance_eta_prod_ratio_0_sum'] = plans_df[cols_distance_eta_prod_ratio_0].sum(axis=1)
    plans_df['plan_distance_eta_prod_ratio_0_max'] = plans_df[cols_distance_eta_prod_ratio_0].max(axis=1)
    plans_df['plan_distance_eta_prod_ratio_0_min'] = plans_df[cols_distance_eta_prod_ratio_0].min(axis=1)
    plans_df['plan_distance_eta_prod_ratio_0_var'] = plans_df[cols_distance_eta_prod_ratio_0].var(axis=1)
    plans_df['plan_distance_eta_prod_ratio_0_skew'] = plans_df[cols_distance_eta_prod_ratio_0].skew(axis=1)

    plans_df['plan_price_distance_eta_prod_ratio_0_mean'] = plans_df[cols_price_distance_eta_prod_ratio_0].mean(axis=1)
    plans_df['plan_price_distance_eta_prod_ratio_0_sum'] = plans_df[cols_price_distance_eta_prod_ratio_0].sum(axis=1)
    plans_df['plan_price_distance_eta_prod_ratio_0_max'] = plans_df[cols_price_distance_eta_prod_ratio_0].max(axis=1)
    plans_df['plan_price_distance_eta_prod_ratio_0_min'] = plans_df[cols_price_distance_eta_prod_ratio_0].min(axis=1)
    plans_df['plan_price_distance_eta_prod_ratio_0_var'] = plans_df[cols_price_distance_eta_prod_ratio_0].var(axis=1)
    plans_df['plan_price_distance_eta_prod_ratio_0_skew'] = plans_df[cols_price_distance_eta_prod_ratio_0].skew(axis=1)

    # rank features
    plans_df[[ c +'_rank' for c in cols_price_distance_ratio]] = plans_df[cols_price_distance_ratio].rank(axis=1)
    plans_df[[ c +'_rank' for c in cols_price_eta_ratio]] = plans_df[cols_price_eta_ratio].rank(axis=1)
    plans_df[[ c +'_rank' for c in cols_distance_eta_ratio]] = plans_df[cols_distance_eta_ratio].rank(axis=1)

    plans_df[[ c +'_rank' for c in cols_price_distance_prod]] = plans_df[cols_price_distance_prod].rank(axis=1)
    plans_df[[ c +'_rank' for c in cols_price_eta_prod]] = plans_df[cols_price_eta_prod].rank(axis=1)
    plans_df[[ c +'_rank' for c in cols_distance_eta_prod]] = plans_df[cols_distance_eta_prod].rank(axis=1)
    plans_df[[ c +'_rank' for c in cols_price_distance_eta_prod]] = plans_df[cols_price_distance_eta_prod].rank(axis=1)

    plans_df[[ c +'_rank' for c in cols_distance_ratio_0]] = plans_df[cols_distance_ratio_0].rank(axis=1)
    plans_df[[ c +'_rank' for c in cols_price_ratio_0]] = plans_df[cols_price_ratio_0].rank(axis=1)
    plans_df[[ c +'_rank' for c in cols_eta_ratio_0]] = plans_df[cols_eta_ratio_0].rank(axis=1)

    plans_df[[ c +'_rank' for c in cols_price_distance_prod_ratio_0]] = plans_df[cols_price_distance_prod_ratio_0].rank(axis=1)
    plans_df[[ c +'_rank' for c in cols_price_eta_prod_ratio_0]] = plans_df[cols_price_eta_prod_ratio_0].rank(axis=1)
    plans_df[[ c +'_rank' for c in cols_distance_eta_prod_ratio_0]] = plans_df[cols_distance_eta_prod_ratio_0].rank(axis=1)
    plans_df[[ c +'_rank' for c in cols_price_distance_eta_prod_ratio_0]] = plans_df[cols_price_distance_eta_prod_ratio_0].rank(axis=1)

    # min-max plan (categorical) for ratio features
    plans_df['plan_price_distance_ratio_max_plan'] = plans_df[cols_price_distance_ratio].idxmax(axis=1).apply(lambda x: x[:6]+'_transport_mode' if type(x)==str else np.nan)
    plans_df['plan_price_eta_ratio_max_plan'] = plans_df[cols_price_eta_ratio].idxmax(axis=1).apply(lambda x: x[:6]+'_transport_mode')
    plans_df['plan_price_distance_ratio_min_plan'] = plans_df[cols_price_distance_ratio].idxmin(axis=1).apply(lambda x: x[:6]+'_transport_mode' if type(x)==str else np.nan)
    plans_df['plan_price_eta_ratio_min_plan'] = plans_df[cols_price_eta_ratio].idxmin(axis=1).apply(lambda x: x[:6]+'_transport_mode' if type(x)==str else np.nan)
    plans_df['plan_distance_eta_ratio_max_plan'] = plans_df[cols_distance_eta_ratio].idxmax(axis=1).apply(lambda x: x[:6]+'_transport_mode' if type(x)==str else np.nan)
    plans_df['plan_distance_eta_ratio_min_plan'] = plans_df[cols_distance_eta_ratio].idxmin(axis=1).apply(lambda x: x[:6]+'_transport_mode' if type(x)==str else np.nan)

    plans_df['plan_price_distance_prod_max_plan'] = plans_df[cols_price_distance_prod].idxmax(axis=1).apply(lambda x: x[:6]+'_transport_mode' if type(x)==str else np.nan)
    plans_df['plan_price_eta_prod_max_plan'] = plans_df[cols_price_eta_prod].idxmax(axis=1).apply(lambda x: x[:6]+'_transport_mode')
    plans_df['plan_price_distance_prod_min_plan'] = plans_df[cols_price_distance_prod].idxmin(axis=1).apply(lambda x: x[:6]+'_transport_mode' if type(x)==str else np.nan)
    plans_df['plan_price_eta_prod_min_plan'] = plans_df[cols_price_eta_prod].idxmin(axis=1).apply(lambda x: x[:6]+'_transport_mode' if type(x)==str else np.nan)
    plans_df['plan_distance_eta_prod_max_plan'] = plans_df[cols_distance_eta_prod].idxmax(axis=1).apply(lambda x: x[:6]+'_transport_mode' if type(x)==str else np.nan)
    plans_df['plan_distance_eta_prod_min_plan'] = plans_df[cols_distance_eta_prod].idxmin(axis=1).apply(lambda x: x[:6]+'_transport_mode' if type(x)==str else np.nan)
    plans_df['plan_price_distance_eta_prod_max_plan'] = plans_df[cols_distance_eta_prod].idxmax(axis=1).apply(lambda x: x[:6]+'_transport_mode' if type(x)==str else np.nan)
    plans_df['plan_price_distance_eta_prod_min_plan'] = plans_df[cols_distance_eta_prod].idxmin(axis=1).apply(lambda x: x[:6]+'_transport_mode' if type(x)==str else np.nan)

    plans_df['plan_distance_ratio_0_max_plan'] = plans_df[cols_distance_ratio_0].idxmax(axis=1).apply(lambda x: x[:6]+'_transport_mode' if type(x)==str else np.nan)
    plans_df['plan_distance_ratio_0_min_plan'] = plans_df[cols_distance_ratio_0].idxmin(axis=1).apply(lambda x: x[:6]+'_transport_mode' if type(x)==str else np.nan)
    plans_df['plan_price_ratio_0_max_plan'] = plans_df[cols_price_ratio_0].idxmax(axis=1).apply(lambda x: x[:6]+'_transport_mode' if type(x)==str else np.nan)
    plans_df['plan_price_ratio_0_min_plan'] = plans_df[cols_price_ratio_0].idxmin(axis=1).apply(lambda x: x[:6]+'_transport_mode' if type(x)==str else np.nan)
    plans_df['plan_eta_ratio_0_max_plan'] = plans_df[cols_eta_ratio_0].idxmax(axis=1).apply(lambda x: x[:6]+'_transport_mode' if type(x)==str else np.nan)
    plans_df['plan_eta_ratio_0_min_plan'] = plans_df[cols_eta_ratio_0].idxmin(axis=1).apply(lambda x: x[:6]+'_transport_mode' if type(x)==str else np.nan)

    plans_df['plan_price_distance_prod_ratio_0_max_plan'] = plans_df[cols_price_distance_prod_ratio_0].idxmax(axis=1).apply(lambda x: x[:6]+'_transport_mode' if type(x)==str else np.nan)
    plans_df['plan_price_distance_prod_ratio_0_min_plan'] = plans_df[cols_price_distance_prod_ratio_0].idxmin(axis=1).apply(lambda x: x[:6]+'_transport_mode' if type(x)==str else np.nan)
    plans_df['plan_price_eta_prod_ratio_0_max_plan'] = plans_df[cols_price_eta_prod_ratio_0].idxmax(axis=1).apply(lambda x: x[:6]+'_transport_mode' if type(x)==str else np.nan)
    plans_df['plan_price_eta_prod_ratio_0_min_plan'] = plans_df[cols_price_eta_prod_ratio_0].idxmin(axis=1).apply(lambda x: x[:6]+'_transport_mode' if type(x)==str else np.nan)
    plans_df['plan_distance_eta_prod_ratio_0_max_plan'] = plans_df[cols_distance_eta_prod_ratio_0].idxmax(axis=1).apply(lambda x: x[:6]+'_transport_mode' if type(x)==str else np.nan)
    plans_df['plan_distance_eta_prod_ratio_0_min_plan'] = plans_df[cols_distance_eta_prod_ratio_0].idxmin(axis=1).apply(lambda x: x[:6]+'_transport_mode' if type(x)==str else np.nan)
    plans_df['plan_price_distance_eta_prod_ratio_0_max_plan'] = plans_df[cols_price_distance_eta_prod_ratio_0].idxmax(axis=1).apply(lambda x: x[:6]+'_transport_mode' if type(x)==str else np.nan)
    plans_df['plan_price_distance_eta_prod_ratio_0_min_plan'] = plans_df[cols_price_distance_eta_prod_ratio_0].idxmin(axis=1).apply(lambda x: x[:6]+'_transport_mode' if type(x)==str else np.nan)

    # map plans
    cols_ratio_plan = ['plan_price_distance_ratio_max_plan','plan_price_distance_ratio_min_plan',
                       'plan_price_eta_ratio_max_plan','plan_price_eta_ratio_min_plan',
                       'plan_distance_eta_ratio_max_plan', 'plan_distance_eta_ratio_min_plan',
                       'plan_price_distance_prod_max_plan', 'plan_price_eta_prod_max_plan',
                       'plan_price_distance_prod_min_plan', 'plan_price_eta_prod_min_plan',
                       'plan_distance_eta_prod_max_plan', 'plan_distance_eta_prod_min_plan',
                       'plan_price_distance_eta_prod_max_plan', 'plan_price_distance_eta_prod_min_plan',
                       'plan_distance_ratio_0_max_plan', 'plan_distance_ratio_0_min_plan',
                       'plan_price_ratio_0_max_plan', 'plan_price_ratio_0_min_plan',
                       'plan_eta_ratio_0_max_plan', 'plan_eta_ratio_0_min_plan',
                       'plan_price_distance_prod_ratio_0_max_plan','plan_price_distance_prod_ratio_0_min_plan',
                       'plan_price_eta_prod_ratio_0_max_plan','plan_price_eta_prod_ratio_0_min_plan',
                       'plan_distance_eta_prod_ratio_0_max_plan', 'plan_distance_eta_prod_ratio_0_min_plan',
                       'plan_price_distance_eta_prod_ratio_0_max_plan','plan_price_distance_eta_prod_ratio_0_min_plan']

    for p in tqdm(cols_ratio_plan):
        for c in cols_transport_mode:
            plans_df[p][plans_df[p]==c] = plans_df[c][plans_df[p]==c]

    # count features
    plans_df['plan_price_distance_ratio_max_plan_count'] = plans_df['plan_price_distance_ratio_max_plan'].map(plans_df['plan_price_distance_ratio_max_plan'].value_counts())
    plans_df['plan_price_distance_ratio_min_plan_count'] = plans_df['plan_price_distance_ratio_min_plan'].map(plans_df['plan_price_distance_ratio_min_plan'].value_counts())
    plans_df['plan_price_eta_ratio_max_plan_count'] = plans_df['plan_price_eta_ratio_max_plan'].map(plans_df['plan_price_eta_ratio_max_plan'].value_counts())
    plans_df['plan_price_eta_ratio_min_plan_count'] = plans_df['plan_price_eta_ratio_min_plan'].map(plans_df['plan_price_eta_ratio_min_plan'].value_counts())
    plans_df['plan_distance_eta_ratio_max_plan_count'] = plans_df['plan_distance_eta_ratio_max_plan'].map(plans_df['plan_distance_eta_ratio_max_plan'].value_counts())
    plans_df['plan_distance_eta_ratio_min_plan_count'] = plans_df['plan_distance_eta_ratio_min_plan'].map(plans_df['plan_distance_eta_ratio_min_plan'].value_counts())

    plans_df['plan_price_distance_prod_max_plan_count'] = plans_df['plan_price_distance_prod_max_plan'].map(plans_df['plan_price_distance_prod_max_plan'].value_counts())
    plans_df['plan_price_distance_prod_min_plan_count'] = plans_df['plan_price_distance_prod_min_plan'].map(plans_df['plan_price_distance_prod_min_plan'].value_counts())
    plans_df['plan_price_eta_prod_max_plan_count'] = plans_df['plan_price_eta_prod_max_plan'].map(plans_df['plan_price_eta_prod_max_plan'].value_counts())
    plans_df['plan_price_eta_prod_min_plan_count'] = plans_df['plan_price_eta_prod_min_plan'].map(plans_df['plan_price_eta_prod_min_plan'].value_counts())
    plans_df['plan_distance_eta_prod_max_plan_count'] = plans_df['plan_distance_eta_prod_max_plan'].map(plans_df['plan_distance_eta_prod_max_plan'].value_counts())
    plans_df['plan_distance_eta_prod_min_plan_count'] = plans_df['plan_distance_eta_prod_min_plan'].map(plans_df['plan_distance_eta_prod_min_plan'].value_counts())
    plans_df['plan_price_distance_eta_prod_max_plan_count'] = plans_df['plan_price_distance_eta_prod_max_plan'].map(plans_df['plan_price_distance_eta_prod_max_plan'].value_counts())
    plans_df['plan_price_distance_eta_prod_min_plan_count'] = plans_df['plan_price_distance_eta_prod_min_plan'].map(plans_df['plan_price_distance_eta_prod_min_plan'].value_counts())

    plans_df['plan_distance_ratio_0_max_plan_count'] = plans_df['plan_distance_ratio_0_max_plan'].map(plans_df['plan_distance_ratio_0_max_plan'].value_counts())
    plans_df['plan_distance_ratio_0_min_plan_count'] = plans_df['plan_distance_ratio_0_min_plan'].map(plans_df['plan_distance_ratio_0_min_plan'].value_counts())
    plans_df['plan_price_ratio_0_max_plan_count'] = plans_df['plan_price_ratio_0_max_plan'].map(plans_df['plan_price_ratio_0_max_plan'].value_counts())
    plans_df['plan_price_ratio_0_min_plan_count'] = plans_df['plan_price_ratio_0_min_plan'].map(plans_df['plan_price_ratio_0_min_plan'].value_counts())
    plans_df['plan_eta_ratio_0_max_plan_count'] = plans_df['plan_eta_ratio_0_max_plan'].map(plans_df['plan_eta_ratio_0_max_plan'].value_counts())
    plans_df['plan_eta_ratio_0_min_plan_count'] = plans_df['plan_eta_ratio_0_min_plan'].map(plans_df['plan_eta_ratio_0_min_plan'].value_counts())

    plans_df['plan_price_distance_prod_ratio_0_max_plan_count'] = plans_df['plan_price_distance_prod_ratio_0_max_plan'].map(plans_df['plan_price_distance_prod_ratio_0_max_plan'].value_counts())
    plans_df['plan_price_distance_prod_ratio_0_min_plan_count'] = plans_df['plan_price_distance_prod_ratio_0_min_plan'].map(plans_df['plan_price_distance_prod_ratio_0_min_plan'].value_counts())
    plans_df['plan_price_eta_prod_ratio_0_max_plan_count'] = plans_df['plan_price_eta_prod_ratio_0_max_plan'].map(plans_df['plan_price_eta_prod_ratio_0_max_plan'].value_counts())
    plans_df['plan_price_eta_prod_ratio_0_min_plan_count'] = plans_df['plan_price_eta_prod_ratio_0_min_plan'].map(plans_df['plan_price_eta_prod_ratio_0_min_plan'].value_counts())
    plans_df['plan_distance_eta_prod_ratio_0_max_plan_count'] = plans_df['plan_distance_eta_prod_ratio_0_max_plan'].map(plans_df['plan_distance_eta_prod_ratio_0_max_plan'].value_counts())
    plans_df['plan_distance_eta_prod_ratio_0_min_plan_count'] = plans_df['plan_distance_eta_prod_ratio_0_min_plan'].map(plans_df['plan_distance_eta_prod_ratio_0_min_plan'].value_counts())
    plans_df['plan_price_distance_eta_prod_ratio_0_max_plan_count'] = plans_df['plan_price_distance_eta_prod_ratio_0_max_plan'].map(plans_df['plan_price_distance_eta_prod_ratio_0_max_plan'].value_counts())
    plans_df['plan_price_distance_eta_prod_ratio_0_min_plan_count'] = plans_df['plan_price_distance_eta_prod_ratio_0_min_plan'].map(plans_df['plan_price_distance_eta_prod_ratio_0_min_plan'].value_counts())

    # save as pkl
    to_pickles(plans_df, '../features/plans', split_size=5)

    line_notify('{} finished.'.format(sys.argv[0]))

if __name__ == '__main__':
    main()
