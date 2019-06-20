
import gc
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import requests
import pickle

from glob import glob
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from tqdm import tqdm

#==============================================================================
# utils
#==============================================================================

# num folds
NUM_FOLDS = 10

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
                     'x_o_round','y_o_round','x_d_round','y_d_round',#'queries_distance_round',
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
        del df_grouped
        gc.collect()
    return df

# scaling predictions
def scalingPredictions(pred_df):
    cols_pred = pred_df.columns.to_list()
    pred_df['pred_min'] = pred_df[cols_pred].min(axis=1)
    pred_df['pred_max'] = pred_df[cols_pred].max(axis=1)
    for c in cols_pred:
        pred_df[c] = (pred_df[c]-pred_df['pred_min'])/(pred_df['pred_max']-pred_df['pred_min'])

    pred_df['pred_sum'] = pred_df[cols_pred].sum(axis=1)
    for c in cols_pred:
        pred_df[c] = pred_df[c]/pred_df['pred_sum']

    return pred_df[cols_pred]

# get best multiple
def getBestMultiple(pred_df, col, cols_pred, output):
    best_f1=0.0
    best_m = 1.0
    f1s = []
    for _m in np.arange(1.0,5.0,0.01):
        tmp_pred = pred_df[cols_pred]
        tmp_pred[col] *= _m
        _f1 = f1_score(pred_df['click_mode'], np.argmax(tmp_pred.values,axis=1),average='weighted')
        f1s.append(_f1)
        print('multiple: {}, f1 score: {}'.format(_m,_f1))
        if _f1 > best_f1:
            best_f1 = _f1
            best_m = _m
        del tmp_pred
    # plot thresholds
    plt.figure()
    plt.plot(np.arange(1.0,5.0,0.01), f1s)
    plt.savefig(output)

    return best_m

# search a best weight for 2 predictions
def getBestWeights(act, pred_lgbm, pred_xgb, output):
    search_range = np.arange(0.3, 0.6, 0.005)
    best_f1=0.0
    best_w = 0.5
    f1s = []
    for _w in search_range:
        # get predictions for each class
        cols_pred=[]
        _pred = pd.DataFrame()
        for i in range(0,12):
            _pred['pred_{}'.format(i)] = _w*pred_lgbm['pred_lgbm_plans{}'.format(i)]+ (1.0-_w)*pred_xgb['pred_xgb_plans{}'.format(i)]
            cols_pred.append('pred_{}'.format(i))

        # calc f1 score
        _f1 = f1_score(act, np.argmax(_pred[cols_pred].values,axis=1),average='weighted')
        f1s.append(_f1)
        print('w: {}, f1 score: {}'.format(_w,_f1))
        if _f1 > best_f1:
            best_f1 = _f1
            best_w = _w
        del _pred

    # plot thresholds
    plt.figure()
    plt.plot(search_range, f1s)
    plt.savefig(output)

    return best_w

# reduce memory usage
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage(deep=True).sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage(deep=True).sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df
