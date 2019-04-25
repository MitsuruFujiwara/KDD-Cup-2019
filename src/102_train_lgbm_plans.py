
import gc
import json
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import time
import warnings

from contextlib import contextmanager
from glob import glob
from pandas.core.common import SettingWithCopyWarning
from sklearn.model_selection import KFold, StratifiedKFold
from tqdm import tqdm

from utils import line_notify, loadpkl
from utils import NUM_FOLDS, FEATS_EXCLUDED, CAT_COLS

#==============================================================================
# Traing LightGBM (only plans, binary classification)
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
        folds = StratifiedKFold(n_splits= num_folds, shuffle=True, random_state=47)
    else:
        folds = KFold(n_splits= num_folds, shuffle=True, random_state=47)

    # Create arrays and dataframes to store results
    oof_preds = np.zeros(train_df.shape[0])
    sub_preds = np.zeros(test_df.shape[0])
    feature_importance_df = pd.DataFrame()
    feats = [f for f in train_df.columns if f not in FEATS_EXCLUDED]

    # k-fold
    for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_df[feats], train_df['target'])):
        train_x, train_y = train_df[feats].iloc[train_idx], train_df['target'].iloc[train_idx]
        valid_x, valid_y = train_df[feats].iloc[valid_idx], train_df['target'].iloc[valid_idx]

        # set data structure
        lgb_train = lgb.Dataset(train_x,
                                label=train_y,
#                                categorical_feature=CAT_COLS,
                                free_raw_data=False)
        lgb_test = lgb.Dataset(valid_x,
                               label=valid_y,
#                               categorical_feature=CAT_COLS,
                               free_raw_data=False)

        # params
        params ={
                'device' : 'gpu',
#                'gpu_use_dp':True,
                'task': 'train',
                'boosting': 'gbdt',
                'objective': 'binary',
                'metric': 'auc',
                'learning_rate': 0.01,
#                'num_leaves': 32,
#                'colsample_bytree': 0.20461151519044,
#                'subsample': 0.805742797052828,
#                'max_depth': 10,
#                'reg_alpha': 0.196466392224054,
#                'reg_lambda': 0.045887453950229,
#                'min_split_gain': 0.247050274075659,
#                'min_child_weight': 23.9202696807894,
#                'min_data_in_leaf': 24,
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
                        num_boost_round=10000,
                        early_stopping_rounds= 200,
                        verbose_eval=100
                        )

        # save model
        clf.save_model('../output/lgbm_'+str(n_fold)+'_binary.txt')

        oof_preds[valid_idx] = clf.predict(valid_x, num_iteration=clf.best_iteration)
        print(test_df[feats])
        sub_preds += clf.predict(test_df[feats], num_iteration=clf.best_iteration) / folds.n_splits

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = feats
        fold_importance_df["importance"] = np.log1p(clf.feature_importance(importance_type='gain', iteration=clf.best_iteration))
        fold_importance_df["fold"] = n_fold + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
        print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(valid_y, oof_preds[valid_idx])))
        del clf, train_x, train_y, valid_x, valid_y
        gc.collect()

    # Full RMSEスコアの表示&LINE通知
    full_auc = roc_auc_score(train_df['outliers'], oof_preds)
    line_notify('Full AUC score %.6f' % full_auc)

    # display importances
    display_importances(feature_importance_df,
                        '../output/lgbm_importances_binary.png',
                        '../output/feature_importance_lgbm_binary.csv')

    if not debug:
        # out of foldの予測値を保存
        train_df.loc[:,'Outlier_Likelyhood'] = oof_preds
        q_train = train_df['Outlier_Likelyhood'].quantile(.98907) # 1.0930%
        train_df.loc[:,'outliers_pred']=train_df['Outlier_Likelyhood'].apply(lambda x: 1 if x > q_train else 0)
        train_df.loc[train_df['outliers_pred']==1,'OOF_PRED']=-33.21928095

        # 提出データの予測値を保存
        test_df.loc[:,'Outlier_Likelyhood'] = sub_preds
        q_test = test_df['Outlier_Likelyhood'].quantile(.9) # 1.0930%
        test_df.loc[:,'outliers']=test_df['Outlier_Likelyhood'].apply(lambda x: 1 if x > q_test else 0) # trainのthreshold使います
        test_df.loc[test_df['outliers']==1,'target']=-33.21928095

        print('q_train: {}, q_test: {}'.format(q_train, q_test))

        # merge
        df = train_df.append(test_df)

        del train_df, test_df
        gc.collect()

        # save as feather
        to_feature(df[['outliers','Outlier_Likelyhood']], '../features')

def main(debug=False):
    with timer("Load Datasets"):
        # load feathers
        df = loadpkl('../features/plans.pkl')

        # split train & test
        train_df = df[df['target'].notnull()]
        test_df = df[df['target'].isnull()]
        del df
        gc.collect()

        if debug:
            train_df=train_df[:10000]

    with timer("Run LightGBM with kfold"):
        kfold_lightgbm(train_df, test_df, num_folds=NUM_FOLDS, stratified=True, debug=debug)

if __name__ == "__main__":
    submission_file_name = "../output/submission_lgbm.csv"
    oof_file_name = "../output/oof_lgbm.csv"
    with timer("Full model run"):
        main(debug=True)