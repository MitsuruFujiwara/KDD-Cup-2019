
import gc
import numpy as np
import pandas as pd
import sys
import warnings

from sklearn.metrics import f1_score

from utils import line_notify, loadpkl, scalingPredictions, getBestMultiple, getBestWeights

#==============================================================================
# Blending
#==============================================================================

warnings.filterwarnings('ignore')

def main():
    # load predictions
    pred_lgbm = loadpkl('../features/lgbm_pred.pkl')
    pred_xgb = loadpkl('../features/xgb_pred.pkl')
    plans = loadpkl('../features/plans.pkl')

    # define columns name list
    cols_pred_lgbm = ['pred_lgbm_plans{}'.format(i) for i in range(0,12)]
    cols_pred_xgb = ['pred_xgb_plans{}'.format(i) for i in range(0,12)]
    cols_transport_mode = ['plan_{}_transport_mode'.format(i) for i in range(0,7)]

    # merge plans & pred
    pred = pred_lgbm[['sid','click_mode']]
    pred = pd.merge(pred, plans[cols_transport_mode+['sid','plan_num_plans']],on='sid', how='left')

    del plans
    gc.collect()

    # scaling predictions
    pred_lgbm[cols_pred_lgbm] = scalingPredictions(pred_lgbm[cols_pred_lgbm])
    pred_xgb[cols_pred_xgb] = scalingPredictions(pred_xgb[cols_pred_xgb])

    # reset index
    pred_lgbm.reset_index(inplace=True,drop=True)
    pred_xgb.reset_index(inplace=True,drop=True)

    # fill predictions for non-exist plans as zero
    for i in range(1,12):
        tmp = np.zeros(len(pred))
        for c in cols_transport_mode:
            tmp += (pred[c]==i).astype(int)
        pred_lgbm['pred_lgbm_plans{}'.format(i)]=pred_lgbm['pred_lgbm_plans{}'.format(i)]*(tmp>0)
        pred_xgb['pred_xgb_plans{}'.format(i)]=pred_xgb['pred_xgb_plans{}'.format(i)]*(tmp>0)

    # get best weight for lgbm & xgboost
    oof_pred_lgbm = pred_lgbm[pred_lgbm.click_mode.notnull()]
    oof_pred_xgb = pred_xgb[pred_xgb.click_mode.notnull()]

    w = getBestWeights(oof_pred_lgbm.click_mode, oof_pred_lgbm, oof_pred_xgb, '../imp/weight.png')

    # calc prediction for each class
    cols_pred =[]
    for i in range(0,12):
        pred['pred_{}'.format(i)] = w*pred_lgbm['pred_lgbm_plans{}'.format(i)]+ (1.0-w)*pred_xgb['pred_xgb_plans{}'.format(i)]
        cols_pred.append('pred_{}'.format(i))

    # get out of fold values
    oof_pred = pred[pred['click_mode'].notnull()]

    # get best multiples
    m4 = getBestMultiple(oof_pred,'pred_4',cols_pred,'../imp/multiple4.png')
    pred['pred_4'] *= m4
    oof_pred['pred_4'] *= m4

    m0 = getBestMultiple(oof_pred,'pred_0',cols_pred,'../imp/multiple0.png')
    pred['pred_0'] *= m0
    oof_pred['pred_0'] *= m0

    m3 = getBestMultiple(oof_pred,'pred_3',cols_pred,'../imp/multiple3.png')
    pred['pred_3'] *= m3
    oof_pred['pred_3'] *= m3

    m6 = getBestMultiple(oof_pred,'pred_6',cols_pred,'../imp/multiple6.png')
    pred['pred_6'] *= m6
    oof_pred['pred_6'] *= m6

    # get recommend mode
    pred['recommend_mode'] = np.argmax(pred[cols_pred].values,axis=1)

    # if number of plans = 1 and recommend mode != 0, set recommend mode as plan 0 mode.
    pred['recommend_mode'][(pred['plan_num_plans']==1)&(pred['recommend_mode']!=0)] = pred['plan_0_transport_mode'][(pred['plan_num_plans']==1)&(pred['recommend_mode']!=0)]

    # split train & test
    sub_pred = pred[pred['click_mode'].isnull()]
    oof_pred = pred[pred['click_mode'].notnull()]

    # out of fold score
    oof_f1_score = f1_score(oof_pred['click_mode'], oof_pred['recommend_mode'],average='weighted')

    # save csv
    oof_pred[['sid','click_mode','recommend_mode']].to_csv(oof_file_name, index=False)
    sub_pred[['sid','recommend_mode']].to_csv(submission_file_name, index=False)

    # line notify
    line_notify('{} finished. f1 score: {}'.format(sys.argv[0],oof_f1_score))

if __name__ == '__main__':
    submission_file_name = '../output/submission_blend.csv'
    oof_file_name = '../output/oof_blend.csv'
    main()
