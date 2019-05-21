
import datetime
import feather
import gc
import json
import pandas as pd
import numpy as np
import warnings

from utils import loadpkl, to_feature, to_json, removeCorrelatedVariables, removeMissingVariables

warnings.filterwarnings('ignore')

#==============================================================================
# Aggregation
#==============================================================================

def main(num_rows=None):
    # load pkls
    df = loadpkl('../features/queries.pkl')
    plans = loadpkl('../features/plans.pkl')
    profiles = loadpkl('../features/profiles.pkl')
#    queries_pred = loadpkl('../features/queries_pred.pkl')
#    queries_profiles_pred = loadpkl('../features/queries_profiles_pred.pkl')

    # merge
    df = pd.merge(df, plans, on=['sid','click_mode'], how='left')
    df = pd.merge(df, profiles, on='pid', how='left')
#    df = pd.merge(df, queries_pred, on='sid', how='left')
#    df = pd.merge(df, queries_profiles_pred, on='sid', how='left')

    del plans, profiles
    gc.collect()

    # count features
    df['pid_count'] = df['pid'].map(df['pid'].value_counts())

    # time diff
    df['plan_req_time_diff'] = (df['plan_time']-df['req_time']).astype(int)

    # distance ratio
    cols_plan_distance = ['plan_{}_distance'.format(i) for i in range(0,7)]

    for i, c in enumerate(cols_plan_distance):
        df['plan_queries_distance_ratio{}'.format(i)] = df[c] / df['queries_distance']

    """
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

    for i in range(0,12):
        cols = ['pred_queries{}'.format(i),'pred_queries_profiles{}'.format(i)]
        df['pred_mean{}'.format(i)] = df[cols].mean(axis=1)
        df['pred_sum{}'.format(i)] = df[cols].sum(axis=1)
        df['pred_max{}'.format(i)] = df[cols].max(axis=1)
        df['pred_min{}'.format(i)] = df[cols].min(axis=1)
        df['pred_var{}'.format(i)] = df[cols].var(axis=1)
        df['pred_skew{}'.format(i)] = df[cols].skew(axis=1)
    """

    # remove missing variables
    col_missing = removeMissingVariables(df,0.75)
    df.drop(col_missing, axis=1, inplace=True)

    # remove correlated variables
    col_drop = removeCorrelatedVariables(df,0.95)
    df.drop(col_drop, axis=1, inplace=True)

    # save as feather
    to_feature(df, '../features')

    # save feature name list
    features_json = {'features':df.columns.tolist()}
    to_json(features_json,'../features/000_all_features.json')

if __name__ == '__main__':
    main()
