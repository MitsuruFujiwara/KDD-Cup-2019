
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
    df = loadpkl('../features/plans.pkl')
    queries = loadpkl('../features/queries.pkl')
    profiles = loadpkl('../features/profiles.pkl')
    queries_pred = loadpkl('../features/queries_pred.pkl')
    queries_profiles_pred = loadpkl('../features/queries_profiles_pred.pkl')
#    plans_pred = loadpkl('../features/plans_pred.pkl')

    # merge
    df = pd.merge(df, queries, on=['sid','click_mode'], how='left')
    df = pd.merge(df, queries_pred, on='sid', how='left')
    df = pd.merge(df, queries_profiles_pred, on='sid', how='left')
    df = pd.merge(df, profiles, on='pid', how='left')
#    df = pd.merge(df, plans_pred, on='sid', how='left')

    del queries, profiles, queries_pred, queries_profiles_pred
    gc.collect()

    # count features
    df['pid_count'] = df['pid'].map(df['pid'].value_counts())

    # isnull feature
#    df['pid_isnull']=df['pid'].isnull().astype(int)

    # time diff
    df['plan_req_time_diff'] = (df['plan_time']-df['req_time']).astype(int)

    # target encoding
    """
    train_df = df[df['click_mode'].notnull()]
    target_dummies=pd.get_dummies(train_df.click_mode.astype(int), prefix='target')
    cols_dummies = target_dummies.columns.to_list()
    train_df = pd.concat([train_df, target_dummies],axis=1)
    df_g = train_df[['pid']+cols_dummies].groupby('pid').mean()
    for i,d in enumerate(cols_dummies):
        df['pid_target_{}'.format(i)]=df['pid'].map(df_g[d])
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
