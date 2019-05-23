
import gc
import numpy as np
import pandas as pd
import sys
import warnings

from sklearn.metrics import f1_score

from utils import line_notify, loadpkl

#==============================================================================
# Blending
#==============================================================================

warnings.filterwarnings('ignore')

def main():
    # load predictions
    pred_lgbm = loadpkl('../features/lgbm_pred.pkl')
    pred_xgb = loadpkl('../features/xgb_pred.pkl')

    # blend predictions
    pred = pred_lgbm[['sid','click_mode']]

    cols_pred =[]
    for i in range(0,12):
        pred['pred_{}'.format(i)] = 0.5*pred_lgbm['pred_lgbm_plans{}'.format(i)]+ 0.5*pred_xgb['pred_xgb_plans{}'.format(i)]
        cols_pred.append('pred_{}'.format(i))

    pred['recommend_mode'] = np.argmax(pred[cols_pred].values,axis=1)

    # add sid & click_mode
    pred['sid'] = pred_lgbm['sid']
    pred['click_mode'] = pred_lgbm['click_mode']

    # split train & test
    oof_pred = pred[pred['click_mode'].notnull()]
    sub_pred = pred[pred['click_mode'].isnull()]

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
