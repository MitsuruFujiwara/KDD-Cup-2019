
import json
import pandas as pd
import numpy as np
import requests
import pickle

from sklearn.metrics import f1_score

#==============================================================================
# utils
#==============================================================================

# num folds
NUM_FOLDS = 5

# features excluded
FEATS_EXCLUDED = ['sid', 'target', 'click_mode', 'plan_time']

# categorical columns
CAT_COLS = ['weekday', 'hour', 'transport_mode']

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

# save pkl
def save2pkl(path, df):
    f = open(path, 'wb')
    pickle.dump(df, f)
    f.close

# load pkl
def loadpkl(path):
    f = open(path, 'rb')
    out = pickle.load(f)
    return out

# make dir
def mkdir_p(path):
    try:
        os.stat(path)
    except:
        os.mkdir(path)

# save multi-pkl files
def to_pickles(df, path, split_size=3):
    """
    path = '../output/mydf'
    wirte '../output/mydf/0.p'
          '../output/mydf/1.p'
          '../output/mydf/2.p'
    """
    print('shape: {}'.format(df.shape))

    gc.collect()
    mkdir_p(path)

    kf = KFold(n_splits=split_size, random_state=326)
    for i, (train_index, val_index) in enumerate(tqdm(kf.split(df))):
        df.iloc[val_index].to_pickle(path+'/'+str(i)+'.pkl')
    return

# read multi-pkl files
def read_pickles(path, col=None, use_tqdm=True):
    if col is None:
        if use_tqdm:
            df = pd.concat([ pd.read_pickle(f) for f in tqdm(sorted(glob(path+'/*'))) ])
        else:
            print('reading {}'.format(path))
            df = pd.concat([ pd.read_pickle(f) for f in sorted(glob(path+'/*')) ])
    else:
        df = pd.concat([ pd.read_pickle(f)[col] for f in tqdm(sorted(glob(path+'/*'))) ])
    return df

# eval function
def eval_f(y_pred, train_data):
    y_true = train_data.label
    y_pred = y_pred.reshape((12, -1)).T
    y_pred = np.argmax(y_pred, axis=1)
    score = f1_score(y_true, y_pred, average='weighted')
    return 'weighted-f1-score', score, True

# load json
def loadJSON(val,key):
    val = json.loads(val)
    return [v[key] for v in val]

# flatten data
def FlattenData(df, key):
    tmp_df = pd.DataFrame(list(df[key]))
    cols = tmp_df.columns.tolist()
    tmp_df['sid'] = df['sid']
    res_df = pd.DataFrame()
    for c in cols:
        _tmp_df = tmp_df[['sid',int(c)]]
        _tmp_df.columns = ['sid',key]
        res_df=res_df.append(_tmp_df)
        del _tmp_df
    res_df.set_index('sid', inplace=True)
    return res_df
