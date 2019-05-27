
import gc
import json
import numpy as np
import optuna
import pandas as pd
import sys
import warnings
import xgboost

from glob import glob
from sklearn.model_selection import KFold, StratifiedKFold
from tqdm import tqdm

from utils import FEATS_EXCLUDED, loadpkl, line_notify, to_json

#==============================================================================
# hyper parameter optimization by optuna
# https://github.com/pfnet/optuna/blob/master/examples/lightgbm_simple.py
#==============================================================================

warnings.filterwarnings('ignore')

# load datasets
CONFIGS = json.load(open('../configs/105_xgb.json'))

# load feathers
FILES = sorted(glob('../features/*.feather'))
DF = pd.concat([pd.read_feather(f) for f in tqdm(FILES, mininterval=60)], axis=1)

# split train & test
TRAIN_DF = DF[DF['click_mode'].notnull()]
del DF
gc.collect()

# use selected features
TRAIN_DF = TRAIN_DF[CONFIGS['features']]

# set card_id as index
TRAIN_DF.set_index('sid', inplace=True)

FEATS = [f for f in TRAIN_DF.columns if f not in FEATS_EXCLUDED]

def objective(trial):
    xgb_train = xgboost.DMatrix(TRAIN_DF[FEATS],
                                  TRAIN_DF['click_mode'])

    param = {
             'device':'gpu',
             'objective':'multi:softmax',
             'tree_method': 'gpu_hist', # GPU parameter
             'predictor': 'gpu_predictor', # GPU parameter
             'eval_metric':'mlogloss',
             'num_class':12,
             'eta': 0.05,
             'booster': 'gbtree',
             'lambda': trial.suggest_loguniform('lambda', 1e-8, 1.0),
             'alpha': trial.suggest_loguniform('alpha', 1e-8, 1.0),
             'silent':1,
             }

    param['gamma'] = trial.suggest_loguniform('gamma', 1e-8, 1.0)
    param['max_depth'] = trial.suggest_int('max_depth', 1, 12)
    param['min_child_weight'] = trial.suggest_uniform('min_child_weight', 0, 45)
    param['subsample']=trial.suggest_uniform('subsample', 0.001, 1)
    param['colsample_bytree']=trial.suggest_uniform('colsample_bytree', 0.001, 1)
    param['colsample_bylevel'] = trial.suggest_uniform('colsample_bylevel', 0.001, 1)

    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=326)

    clf = xgboost.cv(params=param,
                     dtrain=xgb_train,
                     metrics=['mlogloss'],
                     nfold=NUM_FOLDS,
                     folds=list(folds.split(TRAIN_DF[FEATS], TRAIN_DF['click_mode'])),
                     num_boost_round=10000,
                     early_stopping_rounds=200,
                     verbose_eval=100,
                     seed=47
                     )
    gc.collect()
    return clf['test-mlogloss-mean'].iloc[-1]

if __name__ == '__main__':
    study = optuna.create_study()
    study.optimize(objective, n_trials=100)

    print('Number of finished trials: {}'.format(len(study.trials)))

    print('Best trial:')
    trial = study.best_trial

    print('  Value: {}'.format(trial.value))

    print('  Params: ')
    for key, value in trial.params.items():
        print('    {}: {}'.format(key, value))

    # save result
    hist_df = study.trials_dataframe()
    hist_df.to_csv("../output/optuna_result_xgb.csv")

    # save json
    CONFIGS['params'] = trial.params
    to_json(CONFIGS, '../configs/105_xgb.json')

    line_notify('{} finished. Value: {}'.format(sys.argv[0],trial.value))
