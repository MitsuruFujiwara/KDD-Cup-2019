
import datetime
import feather
import gc
import json
import pandas as pd
import numpy as np
import sys
import warnings

from tqdm import tqdm

from utils import loadpkl, read_pickles, to_feature, to_json, line_notify, reduce_mem_usage
from utils import removeCorrelatedVariables, removeMissingVariables, targetEncodingMultiClass

warnings.filterwarnings('ignore')

#==============================================================================
# Aggregation
#==============================================================================

def main(num_rows=None):
    # load pkls
    df = read_pickles('../features/plans')
    queries = loadpkl('../features/queries.pkl')
    profiles = loadpkl('../features/profiles.pkl')
    queries_pred = loadpkl('../features/queries_pred.pkl')
    queries_profiles_pred = loadpkl('../features/queries_profiles_pred.pkl')

    # merge
    df = pd.merge(df, queries, on=['sid','click_mode'], how='left')
    df = pd.merge(df, profiles, on='pid', how='left')
    df = pd.merge(df, queries_pred, on='sid', how='left')
    df = pd.merge(df, queries_profiles_pred, on='sid', how='left')

    del queries, profiles, queries_pred, queries_profiles_pred
    gc.collect()

    # reduce memory usage
    df = reduce_mem_usage(df)

    # count features
    df['pid_count'] = df['pid'].map(df['pid'].value_counts())

    # time diff
    df['plan_req_time_diff'] = (df['plan_time']-df['req_time']).astype(int)

    # distance ratio
    cols_plan_distance = ['plan_{}_distance'.format(i) for i in range(0,7)]

    for i, c in enumerate(cols_plan_distance):
        df['plan_queries_distance_ratio{}'.format(i)] = df[c] / df['queries_distance']
        df['plan_queries_distance_diff{}'.format(i)] = df[c] - df['queries_distance']

    # stats features for preds
    cols_pred_queries = ['pred_queries{}'.format(i) for i in range(0,12)]
    cols_pred_queries_profiles = ['pred_queries_profiles{}'.format(i) for i in range(0,12)]

    df['pred_queries_mean'] = df[cols_pred_queries].mean(axis=1)
    df['pred_queries_sum'] = df[cols_pred_queries].sum(axis=1)
    df['pred_queries_max'] = df[cols_pred_queries].max(axis=1)
    df['pred_queries_min'] = df[cols_pred_queries].min(axis=1)
    df['pred_queries_var'] = df[cols_pred_queries].var(axis=1)
    df['pred_queries_skew'] = df[cols_pred_queries].skew(axis=1)

    df['pred_queries_profiles_mean'] = df[cols_pred_queries_profiles].mean(axis=1)
    df['pred_queries_profiles_sum'] = df[cols_pred_queries_profiles].sum(axis=1)
    df['pred_queries_profiles_max'] = df[cols_pred_queries_profiles].max(axis=1)
    df['pred_queries_profiles_min'] = df[cols_pred_queries_profiles].min(axis=1)
    df['pred_queries_profiles_var'] = df[cols_pred_queries_profiles].var(axis=1)
    df['pred_queries_profiles_skew'] = df[cols_pred_queries_profiles].skew(axis=1)

    # stats features for each classes
    print('stats features...')
    for i in tqdm(range(0,12)):
        cols = ['pred_queries{}'.format(i),'pred_queries_profiles{}'.format(i)]
        df['pred_mean{}'.format(i)] = df[cols].mean(axis=1)
        df['pred_sum{}'.format(i)] = df[cols].sum(axis=1)
        df['pred_max{}'.format(i)] = df[cols].max(axis=1)
        df['pred_min{}'.format(i)] = df[cols].min(axis=1)
        df['pred_var{}'.format(i)] = df[cols].var(axis=1)
        df['pred_skew{}'.format(i)] = df[cols].skew(axis=1)

        cols_target = [c for c in df.columns if '_target_{}'.format(i) in c]
        df['target_mean{}'.format(i)] = df[cols_target].mean(axis=1)
        df['target_sum{}'.format(i)] = df[cols_target].sum(axis=1)
        df['target_max{}'.format(i)] = df[cols_target].max(axis=1)
        df['target_min{}'.format(i)] = df[cols_target].min(axis=1)
        df['target_var{}'.format(i)] = df[cols_target].var(axis=1)
        df['target_skew{}'.format(i)] = df[cols_target].skew(axis=1)

    # post processing
    cols_transport_mode = ['plan_{}_transport_mode'.format(i) for i in range(0,7)]
    print('post processing...')
    for i in tqdm(range(1,12)):
        tmp = np.zeros(len(df))
        for c in cols_transport_mode:
            tmp += (df[c]==i).astype(int)

        cols_target = [c for c in df.columns if '_target_{}'.format(i) in c]
        for c in cols_target+['pred_queries{}'.format(i),'pred_queries_profiles{}'.format(i)]:
            df[c]=df[c]*(tmp>0)

    # reduce memory usage
    df = reduce_mem_usage(df)

    # split data by city
    df1 = df[df['y_o']>37.5]
    df2 = df[df['y_o']<27.5]
    df3 = df[df['x_o']>120.0]

    del df
    gc.collect()

    # cols for target encoding
    cols_target_encoding = ['plan_weekday','plan_hour','plan_is_holiday',
                            'plan_weekday_hour','plan_is_holiday_hour',
                            'plan_num_plans', 'plan_num_free_plans',
                            'x_o_round','y_o_round','x_d_round','y_d_round','queries_distance_round']

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

    cols_min_max_plan = ['plan_distance_max_plan','plan_distance_min_plan',
                         'plan_price_max_plan', 'plan_price_min_plan',
                         'plan_eta_max_plan', 'plan_eta_min_plan']

    cols_transport_mode = ['plan_{}_transport_mode'.format(i) for i in range(0,7)]

    cols_target_encoding = cols_target_encoding + cols_ratio_plan + cols_min_max_plan + cols_transport_mode +['profile_k_means']

    # target encoding for each cities
    print('traget encoding...')
    for i, df in tqdm(enumerate([df1,df2,df3])):

        # target encoding
        df = targetEncodingMultiClass(df, 'click_mode', cols_target_encoding)

        # change dtype
        for col in df.columns.tolist():
            if df[col].dtypes == 'float16':
                df[col] = df[col].astype(np.float32)

        # remove missing variables
        col_missing = removeMissingVariables(df,0.75)
        df.drop(col_missing, axis=1, inplace=True)

        # remove correlated variables
        col_drop = removeCorrelatedVariables(df,0.95)
        df.drop(col_drop, axis=1, inplace=True)

        # save as feather
        to_feature(df, '../features/feats{}'.format(i+1))

        # save feature name list
        features_json = {'features':df.columns.tolist()}
        to_json(features_json,'../features/00{}_all_features.json'.format(i+1))

        del df
        gc.collect()

    line_notify('{} finished.'.format(sys.argv[0]))

if __name__ == '__main__':
    main()
