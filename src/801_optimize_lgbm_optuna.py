
import gc
import json
import lightgbm
import numpy as np
import optuna
import pandas as pd

from glob import glob
from sklearn.model_selection import KFold, StratifiedKFold
from tqdm import tqdm

from utils import FEATS_EXCLUDED, loadpkl, line_notify

#==============================================================================
# hyper parameter optimization by optuna
# https://github.com/pfnet/optuna/blob/master/examples/lightgbm_simple.py
#==============================================================================

# load datasets
CONFIGS = json.load(open('../configs/101_lgbm.json'))

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
    lgbm_train = lightgbm.Dataset(TRAIN_DF[FEATS],
                                  TRAIN_DF['click_mode'],
                                  free_raw_data=False
                                  )

    params = {'objective': 'multiclass',
              'metric': 'multiclass',
              'verbosity': -1,
              'learning_rate': 0.01,
              'num_class': 12,
              'device': 'gpu',
              'boosting_type': 'gbdt',
              'num_leaves': trial.suggest_int('num_leaves', 16, 64),
              'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.001, 1),
              'subsample': trial.suggest_uniform('subsample', 0.001, 1),
              'max_depth': trial.suggest_int('max_depth', 1, 12),
              'reg_alpha': trial.suggest_uniform('reg_alpha', 0, 10),
              'reg_lambda': trial.suggest_uniform('reg_lambda', 0, 10),
              'min_split_gain': trial.suggest_uniform('min_split_gain', 0, 10),
              'min_child_weight': trial.suggest_uniform('min_child_weight', 0, 45),
              'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 16, 64)
              }

    if params['boosting_type'] == 'dart':
        params['drop_rate'] = trial.suggest_loguniform('drop_rate', 1e-8, 1.0)
        params['skip_drop'] = trial.suggest_loguniform('skip_drop', 1e-8, 1.0)
    if params['boosting_type'] == 'goss':
        params['top_rate'] = trial.suggest_uniform('top_rate', 0.0, 1.0)
        params['other_rate'] = trial.suggest_uniform('other_rate', 0.0, 1.0 - params['top_rate'])

    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=326)

    eval_dict = lightgbm.cv(params=params,
                                  train_set=lgbm_train,
                                  metrics=['multiclass'],
                                  nfold=5,
                                  folds=folds.split(TRAIN_DF[FEATS], TRAIN_DF['click_mode']),
                                  num_boost_round=10000,
                                  early_stopping_rounds=200,
                                  verbose_eval=100,
                                  seed=326,
                                 )
    gc.collect()
    return eval_dict['multi_logloss-mean'][-1]

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
    hist_df.to_csv("../output/optuna_result_lgbm.csv")

    # save json
    CONFIGS['params'] = trial.params
    to_json(CONFIGS, '../configs/101_lgbm.json')

    line_notify('optuna LightGBM finished. Value: {}'.format(trial.value))
