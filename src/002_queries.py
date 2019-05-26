
import gc
import json
import pandas as pd
import numpy as np
import sys
import warnings

from chinese_calendar import is_holiday
from sklearn.decomposition import TruncatedSVD

from utils import save2pkl, to_json, line_notify, targetEncodingMultiClass

warnings.filterwarnings('ignore')

#==============================================================================
# Preprocessing Queries
#==============================================================================

def main(num_rows=None):
    # load csv
    train_queries = pd.read_csv('../input/data_set_phase1/train_queries.csv',nrows=num_rows)
    test_queries = pd.read_csv('../input/data_set_phase1/test_queries.csv',nrows=num_rows)
    train_clicks = pd.read_csv('../input/data_set_phase1/train_clicks.csv')

    # merge click
    train_queries = pd.merge(train_queries, train_clicks[['sid','click_mode']], on='sid', how='left')

    # fill na (no click)
    train_queries['click_mode'].fillna(0, inplace=True)

    # set test target as nan
    test_queries['click_mode'] = np.nan

    # merge train & test
    queries_df = train_queries.append(test_queries)

    del train_queries, test_queries
    gc.collect()

    # to datetime
    queries_df['req_time'] = pd.to_datetime(queries_df['req_time'])

    # distance features
    queries_df['x_o']=queries_df['o'].apply(lambda x: x.split(',')[0]).astype(float)
    queries_df['y_o']=queries_df['o'].apply(lambda x: x.split(',')[1]).astype(float)
    queries_df['x_d']=queries_df['d'].apply(lambda x: x.split(',')[0]).astype(float)
    queries_df['y_d']=queries_df['d'].apply(lambda x: x.split(',')[1]).astype(float)

    # count features
    queries_df['queries_o_count']=queries_df['o'].map(queries_df['o'].value_counts())
    queries_df['queries_d_count']=queries_df['d'].map(queries_df['d'].value_counts())

    queries_df['queries_x_o_count']=queries_df['x_o'].map(queries_df['x_o'].value_counts())
    queries_df['queries_y_o_count']=queries_df['y_o'].map(queries_df['y_o'].value_counts())
    queries_df['queries_x_d_count']=queries_df['x_d'].map(queries_df['x_d'].value_counts())
    queries_df['queries_y_d_count']=queries_df['y_d'].map(queries_df['y_d'].value_counts())

    queries_df['queries_distance'] = np.sqrt((queries_df['x_o']-queries_df['x_d'])**2 + (queries_df['y_o']-queries_df['y_d'])**2)

    queries_df['o_d'] = queries_df['o'].astype(str)+'_'+queries_df['d'].astype(str)
    queries_df['queries_o_d_count'] = queries_df['o_d'].map(queries_df['o_d'].value_counts())

    # datetime features
    queries_df['queries_weekday'] = queries_df['req_time'].dt.weekday
    queries_df['queries_hour'] = queries_df['req_time'].dt.hour
    queries_df['queries_is_holiday'] = queries_df['req_time'].apply(lambda x: is_holiday(x)).astype(int)

    queries_df['queries_weekday_count'] = queries_df['queries_weekday'].map(queries_df['queries_weekday'].value_counts())
    queries_df['queries_hour_count'] = queries_df['queries_hour'].map(queries_df['queries_hour'].value_counts())

    # coordinate & datetime features
    queries_df['o_d_is_holiday'] = queries_df['queries_is_holiday'].astype(str)+'_'+queries_df['o_d']
    queries_df['o_d_weekday'] = queries_df['queries_weekday'].astype(str)+'_'+queries_df['o_d']
    queries_df['o_d_hour'] = queries_df['queries_hour'].astype(str)+'_'+queries_df['o_d']

    queries_df['o_is_holiday'] = queries_df['queries_is_holiday'].astype(str)+'_'+queries_df['o']
    queries_df['o_weekday'] = queries_df['queries_weekday'].astype(str)+'_'+queries_df['o']
    queries_df['o_hour'] = queries_df['queries_hour'].astype(str)+'_'+queries_df['o']

    queries_df['d_is_holiday'] = queries_df['queries_is_holiday'].astype(str)+'_'+queries_df['d']
    queries_df['d_weekday'] = queries_df['queries_weekday'].astype(str)+'_'+queries_df['d']
    queries_df['d_hour'] = queries_df['queries_hour'].astype(str)+'_'+queries_df['d']

    queries_df['queries_o_d_is_holiday_count'] = queries_df['o_d_is_holiday'].map(queries_df['o_d_is_holiday'].value_counts())
    queries_df['queries_o_d_weekday_count'] = queries_df['o_d_weekday'].map(queries_df['o_d_weekday'].value_counts())
    queries_df['queries_o_d_hour_count'] = queries_df['o_d_hour'].map(queries_df['o_d_hour'].value_counts())

    queries_df['queries_o_is_holiday_count'] = queries_df['o_d_is_holiday'].map(queries_df['o_d_is_holiday'].value_counts())
    queries_df['queries_o_weekday_count'] = queries_df['o_d_weekday'].map(queries_df['o_d_weekday'].value_counts())
    queries_df['queries_o_hour_count'] = queries_df['o_d_hour'].map(queries_df['o_d_hour'].value_counts())

    queries_df['queries_o_d_is_holiday_count'] = queries_df['o_d_is_holiday'].map(queries_df['o_d_is_holiday'].value_counts())
    queries_df['queries_o_d_weekday_count'] = queries_df['o_d_weekday'].map(queries_df['o_d_weekday'].value_counts())
    queries_df['queries_o_d_hour_count'] = queries_df['o_d_hour'].map(queries_df['o_d_hour'].value_counts())

    # rounded value features
    queries_df['x_o_round'] = queries_df['x_o'].round(1)
    queries_df['y_o_round'] = queries_df['y_o'].round(1)
    queries_df['x_d_round'] = queries_df['x_d'].round(1)
    queries_df['y_d_round'] = queries_df['y_d'].round(1)
    queries_df['queries_distance_round'] = queries_df['queries_distance'].round(1)

    queries_df['o_round'] = queries_df['x_o_round'].astype(str)+'_'+queries_df['y_o_round'].astype(str)
    queries_df['d_round'] = queries_df['x_d_round'].astype(str)+'_'+queries_df['y_d_round'].astype(str)
    queries_df['o_d_round'] = queries_df['o_round'].astype(str)+'_'+queries_df['d_round'].astype(str)

    queries_df['queries_x_o_round_count'] = queries_df['x_o_round'].map(queries_df['x_o_round'].value_counts())
    queries_df['queries_y_o_round_count'] = queries_df['y_o_round'].map(queries_df['y_o_round'].value_counts())
    queries_df['queries_x_d_round_count'] = queries_df['x_d_round'].map(queries_df['x_d_round'].value_counts())
    queries_df['queries_y_d_round_count'] = queries_df['y_d_round'].map(queries_df['y_d_round'].value_counts())
    queries_df['queries_distance_round_count'] = queries_df['queries_distance_round'].map(queries_df['queries_distance_round'].value_counts())
    queries_df['queries_o_round_count'] = queries_df['o_round'].map(queries_df['o_round'].value_counts())
    queries_df['queries_d_round_count'] = queries_df['d_round'].map(queries_df['d_round'].value_counts())
    queries_df['queries_o_d_round_count'] = queries_df['o_d_round'].map(queries_df['o_d_round'].value_counts())

    # factorize
    queries_df['x_o_round'], _ = pd.factorize(queries_df['x_o_round'])
    queries_df['y_o_round'], _ = pd.factorize(queries_df['y_o_round'])
    queries_df['x_d_round'], _ = pd.factorize(queries_df['x_d_round'])
    queries_df['y_d_round'], _ = pd.factorize(queries_df['y_d_round'])
    queries_df['queries_distance_round'], _ = pd.factorize(queries_df['queries_distance_round'])

    # target encoding
    cols_encoding = ['x_o_round','y_o_round','x_d_round','y_d_round','queries_distance_round']
    queries_df = targetEncodingMultiClass(queries_df, 'click_mode', cols_encoding)

    # drop string features
    cols_drop = ['o','d','o_d','o_d_is_holiday','o_d_weekday','o_d_hour',
                 'o_is_holiday','o_weekday','o_hour','d_is_holiday','d_weekday','d_hour',
                 'o_round','d_round','o_d_round']
    queries_df.drop(cols_drop, axis=1, inplace=True)

    # save as pkl
    save2pkl('../features/queries.pkl', queries_df)

    # save configs
    configs = json.load(open('../configs/101_lgbm_queries.json'))
    configs['features'] = queries_df.columns.to_list()
    to_json(configs,'../configs/101_lgbm_queries.json')

    line_notify('{} finished.'.format(sys.argv[0]))

if __name__ == '__main__':
    main()
