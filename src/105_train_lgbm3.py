
import gc
import json
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import sys
import time
import warnings

from contextlib import contextmanager
from glob import glob
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold, StratifiedKFold
from tqdm import tqdm

from utils import line_notify, loadpkl, eval_f, save2pkl
from utils import NUM_FOLDS, FEATS_EXCLUDED, CAT_COLS

#==============================================================================
# Traing LightGBM (city 3)
#==============================================================================

warnings.filterwarnings('ignore')

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))

# Display/plot feature importance
def display_importances(feature_importance_df_, outputpath, csv_outputpath):
    cols = feature_importance_df_[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]

    # for checking all importance
    _feature_importance_df_=feature_importance_df_.groupby('feature').sum()
    _feature_importance_df_.to_csv(csv_outputpath)

    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig(outputpath)

# LightGBM GBDT with KFold or Stratified KFold
def kfold_lightgbm(train_df,test_df,num_folds,stratified=False,debug=False):

    print("Starting LightGBM. Train shape: {}, test shape: {}".format(train_df.shape, test_df.shape))

    # Cross validation model
    if stratified:
        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=326)
    else:
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=326)

    # Create arrays and dataframes to store results
    oof_preds = np.zeros((train_df.shape[0],12))
    sub_preds = np.zeros((test_df.shape[0],12))
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in FEATS_EXCLUDED]

    # k-fold
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['click_mode'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['click_mode'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['click_mode'].iloc[valid_idx]

        # set data structure
        lgb_train = lgb.Dataset(train_x,
                                label=train_y,
                                categorical_feature=CAT_COLS,
                                free_raw_data=False)
        lgb_test = lgb.Dataset(valid_x,
                               label=valid_y,
                               categorical_feature=CAT_COLS,
                               free_raw_data=False)

        # params
        params ={
                'device' : 'gpu',
#                'gpu_use_dp':True,
                'task': 'train',
                'boosting': 'gbdt',
                'objective': 'multiclass',
                'metric': 'multiclass',
                'learning_rate': 0.01,
                'num_class': 12,
                'num_leaves': 52,
                'colsample_bytree': 0.3490457769968177,
                'subsample': 0.543646263362097,
                'max_depth': 11,
                'reg_alpha': 4.762312990232561,
                'reg_lambda': 9.98131082276387,
                'min_split_gain': 0.19161156850826594,
                'min_child_weight': 15.042054927368088,
                'min_data_in_leaf': 17,
                'verbose': -1,
                'seed':int(2**n_fold),
                'bagging_seed':int(2**n_fold),
                'drop_seed':int(2**n_fold)
                }

        clf = lgb.train(
                        params,
                        lgb_train,
                        valid_sets=[lgb_train, lgb_test],
                        valid_names=['train', 'test'],
#                        feval=eval_f,
                        num_boost_round=10000,
                        early_stopping_rounds= 200,
                        verbose_eval=100
                        )

        # save model
        clf.save_model('../output/lgbm_3_{}.txt'.format(n_fold))

        oof_preds[valid_idx] = clf.predict(valid_x, num_iteration=clf.best_iteration)
        sub_preds += clf.predict(test_df[feats], num_iteration=clf.best_iteration) / folds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = np.log1p(clf.feature_importance(importance_type='gain', iteration=clf.best_iteration))
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print('Fold %2d F1 Score : %.6f' % (n_fold + 1, f1_score(valid_y,np.argmax(oof_preds[valid_idx],axis=1),average='weighted')))
        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()

    # Full F1 Score & LINE Notify
    full_f1 = f1_score(train_df['click_mode'], np.argmax(oof_preds,axis=1),average='weighted')
    print('Full F1 Score %.6f' % full_f1)
    line_notify('Full F1 Score %.6f' % full_f1)

    # display importances
    display_importances(feature_importance_df,
                        '../imp/lgbm_importances_3.png',
                        '../imp/feature_importance_lgbm_3.csv')

    if not debug:
        # save prediction for submit
        test_df['recommend_mode'] = np.argmax(sub_preds, axis=1)
        test_df = test_df.reset_index()

        # post processing
        test_df['recommend_mode'][(test_df['plan_num_plans']==1)&(test_df['recommend_mode']!=0)] = test_df['plan_0_transport_mode'][(test_df['plan_num_plans']==1)&(test_df['recommend_mode']!=0)]

        # save csv
        test_df[['sid','recommend_mode']].to_csv(submission_file_name, index=False)

        # save out of fold prediction
        train_df.loc[:,'recommend_mode'] = np.argmax(oof_preds, axis=1)
        train_df = train_df.reset_index()
        train_df[['sid','click_mode','recommend_mode']].to_csv(oof_file_name, index=False)

        # save prediction for submit
        sub_preds = pd.DataFrame(sub_preds)
        sub_preds.columns = ['pred_lgbm_plans{}'.format(c) for c in sub_preds.columns]
        sub_preds['sid'] = test_df['sid']
        sub_preds['click_mode'] = test_df['click_mode']

        # save out of fold prediction
        oof_preds = pd.DataFrame(oof_preds)
        oof_preds.columns = ['pred_lgbm_plans{}'.format(c) for c in oof_preds.columns]
        oof_preds['sid'] = train_df['sid']
        oof_preds['click_mode'] = train_df['click_mode']

        # merge
        df = oof_preds.append(sub_preds)

        # save as pkl
        save2pkl('../features/lgbm_pred_3.pkl', df)

        line_notify('{} finished.'.format(sys.argv[0]))

def main(debug=False):
    with timer("Load Datasets"):
        # load feathers
        files = sorted(glob('../features/feats3/*.feather'))
        df = pd.concat([pd.read_feather(f) for f in tqdm(files, mininterval=60)], axis=1)

        # use selected features
        df = df[configs['features']]

        # set card_id as index
        df.set_index('sid', inplace=True)

        # split train & test
        train_df = df[df['click_mode'].notnull()]
        test_df = df[df['click_mode'].isnull()]

        del df
        gc.collect()

        if debug:
            train_df=train_df.iloc[:1000]

    with timer("Run LightGBM with kfold"):
        kfold_lightgbm(train_df, test_df, num_folds=NUM_FOLDS, stratified=True, debug=debug)

if __name__ == "__main__":
    submission_file_name = "../output/submission_lgbm_3.csv"
    oof_file_name = "../output/oof_lgbm_3.csv"
    configs = json.load(open('../configs/105_lgbm.json'))
    with timer("Full model run"):
        main(debug=False)
