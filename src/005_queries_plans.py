
import datetime
import feather
import gc
import json
import pandas as pd
import numpy as np
import sys
import warnings

from utils import loadpkl, to_feature, to_json, save2pkl, line_notify

warnings.filterwarnings('ignore')

#==============================================================================
# Preprocessing Queries & Profiles
#==============================================================================

def main(num_rows=None):
    # load pkls
    queries = loadpkl('../features/queries.pkl')
    plans = loadpkl('../features/plans.pkl')

    # merge
    df = pd.merge(queries, plans, on=['sid','click_mode'], how='left')
    del queries, plans
    gc.collect()

    # save pkl
    save2pkl('../features/queries_plans.pkl', df)

    # save configs
    configs = json.load(open('../configs/103_lgbm_queries_plans.json'))
    configs['features'] = df.columns.to_list()
    to_json(configs,'../configs/103_lgbm_queries_plans.json')

    line_notify('{} finished.'.format(sys.argv[0]))

if __name__ == '__main__':
    main()
