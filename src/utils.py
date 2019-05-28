
import json
import pandas as pd
import numpy as np
import requests
import pickle

from sklearn.metrics import f1_score
from tqdm import tqdm

#==============================================================================
# utils
#==============================================================================

# num folds
NUM_FOLDS = 5

# features excluded
FEATS_EXCLUDED = ['index', 'sid', 'pid', 'click_mode', 'plan_time', 'req_time']

# categorical columns
cat_cols = ['plan_{}_transport_mode'.format(i) for i in range(0,5)]
CAT_COLS = cat_cols+['plan_weekday', 'plan_hour',
                     'plan_distance_max_plan', 'plan_distance_min_plan',
                     'plan_price_max_plan', 'plan_price_min_plan',
                     'plan_eta_max_plan', 'plan_eta_min_plan',
                     'plan_price_distance_ratio_max_plan', 'plan_price_distance_ratio_min_plan',
                     'plan_price_eta_ratio_max_plan', 'plan_price_eta_ratio_min_plan',
                     'plan_distance_eta_ratio_max_plan', 'plan_distance_eta_ratio_min_plan',
                     'plan_price_distance_prod_max_plan', 'plan_price_eta_prod_max_plan',
                     'plan_price_distance_prod_min_plan', 'plan_price_eta_prod_min_plan',
                     'plan_distance_eta_prod_max_plan', 'plan_distance_eta_prod_min_plan',
                     'plan_price_distance_eta_prod_max_plan', 'plan_price_distance_eta_prod_min_plan',
                     'plan_distance_ratio_0_max_plan','plan_distance_ratio_0_min_plan',
                     'plan_price_ratio_0_max_plan', 'plan_price_ratio_0_min_plan',
                     'plan_eta_ratio_0_max_plan', 'plan_eta_ratio_0_min_plan',
                     'plan_price_distance_prod_ratio_0_max_plan','plan_price_distance_prod_ratio_0_min_plan',
                     'plan_price_eta_prod_ratio_0_max_plan','plan_price_eta_prod_ratio_0_min_plan',
                     'plan_distance_eta_prod_ratio_0_max_plan', 'plan_distance_eta_prod_ratio_0_min_plan',
                     'plan_price_distance_eta_prod_ratio_0_max_plan','plan_price_distance_eta_prod_ratio_0_min_plan',
                     'x_o_round','y_o_round','x_d_round','y_d_round','queries_distance_round',
                     'profile_k_means']

# to feather
def to_feature(df, path):
    if df.columns.duplicated().sum()>0:
        raise Exception('duplicated!: {}'.format(df.columns[df.columns.duplicated()]))
    df.reset_index(inplace=True)
    df.columns = [c.replace('/', '-').replace(' ', '-') for c in df.columns]
    for c in df.columns:
        df[[c]].to_feather('{}/{}.feather'.format(path,c))
    return

# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns

# target encoding
def targetEncoding(train_df, col):
    target_dummies = pd.get_dummies(train_df['click_mode'].astype(int), prefix='target')
    train_df = pd.concat([train_df, target_dummies],axis=1)
    for c in target_dummies.columns.to_list:
        dict_for_map = target_dummies.groupby(col)[target].mean()

    res = df[col].map(dict_for_map)
    return res

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

# flatten data simple
def FlattenDataSimple(df, key):
    res_df = pd.DataFrame(list(df[key]))
    res_df.columns = ['plan_{}_{}'.format(c, key) for c in res_df.columns.tolist()]
    res_df['sid'] = df['sid']
    res_df.set_index('sid', inplace=True)
    return res_df

# save json
def to_json(data_dict, path):
    with open(path, 'w') as f:
        json.dump(data_dict, f, indent=4)

# target encoding for multi class
def targetEncodingMultiClass(df, col_target, cols_encoding):
    df_target = df[df[col_target].notnull()]
    df_dummies = pd.get_dummies(df_target[col_target].astype(int), prefix='target')
    cols_dummies = df_dummies.columns.to_list()
    df_target = pd.concat([df_target,df_dummies],axis=1)
    print('target encoding...')
    for c in tqdm(cols_encoding):
        df_grouped = df_target[[c]+cols_dummies].groupby(c).mean()
        for i,d in enumerate(cols_dummies):
            df['{}_target_{}'.format(c,i)] = df[c].map(df_grouped[d])
    return df
