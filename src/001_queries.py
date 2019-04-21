import gc
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

#==============================================================================
# Preprocessing Queries
#==============================================================================

def main(num_rows=None):
    # load csv
    train_queries = pd.read_csv('../input/data_set_phase1/train_queries.csv',nrows=num_rows)
    test_queries = pd.read_csv('../input/data_set_phase1/test_queries.csv',nrows=num_rows)

    # is test
    test_queries['is_test']=True
    train_queries['is_test']=False

    # merge train & test
    queries_df = train_queries.append(test_queries)

    # TODO: Preprocessing

    # save as pkl
    save2pkl('../features/queries.pkl', queries_df)

if __name__ == '__main__':
    main()
