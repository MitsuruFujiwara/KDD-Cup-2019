
import pandas as pd
import numpy as np
import requests

#==============================================================================
# utils
#==============================================================================

NUM_FOLDS = 5

# to feather
def to_feature(df, path):
    if df.columns.duplicated().sum()>0:
        raise Exception('duplicated!: {}'.format(df.columns[df.columns.duplicated()]))
    df.reset_index(inplace=True)
    df.columns = [c.replace('/', '-').replace(' ', '-') for c in df.columns]
    for c in df.columns:
        df[[c]].to_feather('{}/{}.feather'.format(path,c))
    return

# remove correlated variables
def removeCorrelatedVariables(data, threshold):
    print('Removing Correlated Variables...')
    corr_matrix = data.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    col_drop = [column for column in upper.columns if any(upper[column] > threshold) & ('target' not in column)]
    return col_drop

# remove missing variables
def removeMissingVariables(data, threshold):
    print('Removing Missing Variables...')
    missing = (data.isnull().sum() / len(data)).sort_values(ascending = False)
    col_missing = missing.index[missing > threshold]
    col_missing = [column for column in col_missing if 'target' not in column]
    return col_missing

# LINE Notify
def line_notify(message):
    f = open('../input/line_token.txt')
    token = f.read()
    f.close
    line_notify_token = token.replace('\n', '')
    line_notify_api = 'https://notify-api.line.me/api/notify'

    payload = {'message': message}
    headers = {'Authorization': 'Bearer ' + line_notify_token}
    line_notify = requests.post(line_notify_api, data=payload, headers=headers)
    print(message)
