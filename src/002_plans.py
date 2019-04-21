import gc
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

#==============================================================================
# Preprocessing Plans
#==============================================================================

def main(num_rows=None):
    # load csv
    train_plans = pd.read_csv('../input/data_set_phase1/train_plans.csv',nrows=num_rows)
    test_plans = pd.read_csv('../input/data_set_phase1/test_plans.csv',nrows=num_rows)

    # is test
    test_plans['is_test']=True
    train_plans['is_test']=False

    # merge train & test
    plans_df = train_plans.append(test_plans)

    # TODO: Preprocessing

    # save as pkl
    save2pkl('../features/plans.pkl', plans_df)

if __name__ == '__main__':
    main()
