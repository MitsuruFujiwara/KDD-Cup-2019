
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
    train_queries = pd.merge(train_queries, train_clicks[['sid','click_mode']], on='sid', how='outer')

    # merge profiles
    train_queries = pd.merge(train_queries, profiles, on='sid', how='outer')

    # fill na
    train_queries['click_mode'].fillna(0, inplace=True)

    # set test target as nan
    test_queries['click_mode'] = np.nan

    # merge train & test
    queries_df = train_queries.append(test_queries)

    del train_queries, test_queries
    gc.collect()

    # set index
    queries_df.set_index('sid', inplace=True)

    # to datetime
    queries_df['req_time'] = pd.to_datetime(queries_df['req_time'])

    # features distance
    queries_df['x_o']=queries_df['o'].apply(lambda x: x.split(',')[0]).astype(float)
    queries_df['y_o']=queries_df['o'].apply(lambda x: x.split(',')[1]).astype(float)
    queries_df['x_d']=queries_df['d'].apply(lambda x: x.split(',')[0]).astype(float)
    queries_df['y_d']=queries_df['d'].apply(lambda x: x.split(',')[1]).astype(float)

    # TODO: Preprocessing

    # save as pkl
    save2pkl('../features/queries.pkl', queries_df)

if __name__ == '__main__':
    main()
