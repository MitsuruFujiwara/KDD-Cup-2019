
import gc
import lightgbm as lgb
import numpy as np
import pandas as pd
import sys
import warnings
import xgboost as xgb

from sklearn.metrics import f1_score

from utils import line_notify, loadpkl, scalingPredictions, getBestMultiple, getBestWeights, read_pickles

#==============================================================================
# Blending
#==============================================================================

warnings.filterwarnings('ignore')

def main():
    # load predictions
    pred_lgbm1 = loadpkl('../features/lgbm_pred_1.pkl')
    pred_lgbm2 = loadpkl('../features/lgbm_pred_2.pkl')
    pred_lgbm3 = loadpkl('../features/lgbm_pred_3.pkl')
    plans = read_pickles('../features/plans')
    preds = [pred_lgbm1,pred_lgbm2,pred_lgbm3]

    # define columns name list
    cols_pred_lgbm = ['pred_lgbm_plans{}'.format(i) for i in range(0,12)]
    cols_transport_mode = ['plan_{}_transport_mode'.format(i) for i in range(0,7)]

    # remove columns
    cols_drop = [c for c in plans.columns if c not in cols_transport_mode+['sid','plan_num_plans','click_mode']]
    plans.drop(cols_drop,axis=1,inplace=True)

    # postprocessing
    sub_preds = []
    oof_preds = []
    for i, pred_lgbm in enumerate(preds):

        # merge plans & pred
        pred = pred_lgbm[['sid','click_mode']]
        pred = pd.merge(pred, plans[cols_transport_mode+['sid','plan_num_plans']], on='sid', how='left')

        # scaling predictions
        pred_lgbm[cols_pred_lgbm] = scalingPredictions(pred_lgbm[cols_pred_lgbm])

        # reset index
        pred_lgbm.reset_index(inplace=True,drop=True)

        # fill predictions for non-exist plans as zero
        for j in range(1,12):
            tmp = np.zeros(len(pred))
            for c in cols_transport_mode:
                tmp += (pred[c]==j).astype(int)
            pred_lgbm['pred_lgbm_plans{}'.format(j)]=pred_lgbm['pred_lgbm_plans{}'.format(j)]*(tmp>0)

        # get best weight for lgbm & xgboost
        oof_pred_lgbm = pred_lgbm[pred_lgbm['click_mode'].notnull()]

        # calc prediction for each class
        cols_pred =[]
        for j in range(0,12):
            pred['pred_{}'.format(j)] = pred_lgbm['pred_lgbm_plans{}'.format(j)]
            cols_pred.append('pred_{}'.format(j))

        # get out of fold values
        oof_pred = pred[pred['click_mode'].notnull()]

        # get best multiples
        m0 = getBestMultiple(oof_pred,'pred_0',cols_pred,'../imp/multiple0_{}.png'.format(i+1))
        pred['pred_0'] *= m0
        oof_pred['pred_0'] *= m0

        m3 = getBestMultiple(oof_pred,'pred_3',cols_pred,'../imp/multiple3_{}.png'.format(i+1))
        pred['pred_3'] *= m3
        oof_pred['pred_3'] *= m3

        m4 = getBestMultiple(oof_pred,'pred_4',cols_pred,'../imp/multiple4_{}.png'.format(i+1))
        pred['pred_4'] *= m4
        oof_pred['pred_4'] *= m4

        # get recommend mode
        pred['recommend_mode'] = np.argmax(pred[cols_pred].values,axis=1)

        # if number of plans = 1 and recommend mode != 0, fill recommend mode with plan 0 mode.
        pred['recommend_mode'][(pred['plan_num_plans']==1)&(pred['recommend_mode']!=0)] = pred['plan_0_transport_mode'][(pred['plan_num_plans']==1)&(pred['recommend_mode']!=0)]

        # split train & test
        _sub_pred = pred[pred['click_mode'].isnull()]
        _oof_pred = pred[pred['click_mode'].notnull()]

        sub_preds.append(_sub_pred)
        oof_preds.append(_oof_pred)

        del pred, _sub_pred, _oof_pred
        gc.collect()

    # merge preds
    sub_pred = sub_preds[0].append(sub_preds[1])
    sub_pred = sub_pred.append(sub_preds[2])
    sub_pred = pd.merge(plans[plans['click_mode'].isnull()][['sid','click_mode']],sub_pred[['sid','recommend_mode']], on='sid', how='left')

    oof_pred = oof_preds[0].append(oof_preds[1])
    oof_pred = oof_pred.append(oof_preds[2])
    oof_pred = pd.merge(plans[plans['click_mode'].notnull()][['sid','click_mode']],oof_pred[['sid','recommend_mode']], on='sid', how='left')

    del sub_preds, oof_preds, plans

    # out of fold score
    oof_f1_score = f1_score(oof_pred['click_mode'], oof_pred['recommend_mode'],average='weighted')

    # save csv
    oof_pred[['sid','click_mode','recommend_mode']].to_csv(oof_file_name, index=False)
    sub_pred[['sid','recommend_mode']].to_csv(submission_file_name, index=False)

    # line notify
    line_notify('{} finished. f1 score: {}'.format(sys.argv[0],oof_f1_score))

if __name__ == '__main__':
    submission_file_name = '../output/submission_blend_phase2.csv'
    oof_file_name = '../output/oof_blend_phase2.csv'
    main()
