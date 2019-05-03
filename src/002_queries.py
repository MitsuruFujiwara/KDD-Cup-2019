
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

    # features distance
    queries_df['x_o']=queries_df['o'].apply(lambda x: x.split(',')[0]).astype(float)
    queries_df['y_o']=queries_df['o'].apply(lambda x: x.split(',')[1]).astype(float)
    queries_df['x_d']=queries_df['d'].apply(lambda x: x.split(',')[0]).astype(float)
    queries_df['y_d']=queries_df['d'].apply(lambda x: x.split(',')[1]).astype(float)

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
    queries_df['queries_weekday_count'] = queries_df['queries_weekday'].map(queries_df['queries_weekday'].value_counts())
    queries_df['queries_hour_count'] = queries_df['queries_hour'].map(queries_df['queries_hour'].value_counts())

    # TODO: Preprocessing

    # drop string features
    queries_df.drop(['o','d','x_o','y_o','x_d','y_d','o_d'], axis=1, inplace=True)

    # save as pkl
    save2pkl('../features/queries.pkl', queries_df)

if __name__ == '__main__':
    main()
