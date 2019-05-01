
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

    # merge
    df = pd.merge(df, queries, on='sid', how='left')
    df = pd.merge(df, profiles, on='pid', how='left')

    del queries, profiles
    gc.collect()

    # remove missing variables
#    col_missing = removeMissingVariables(df,0.75)
#    df.drop(col_missing, axis=1, inplace=True)

    # remove correlated variables
#    col_drop = removeCorrelatedVariables(df,0.9)
#    df.drop(col_drop, axis=1, inplace=True)

    # save as feather
    to_feature(df, '../features')

    # save feature name list
    features_json = {'features':df.columns.tolist()}
    to_json(features_json,'../features/000_all_features.json')

if __name__ == '__main__':
    main()
