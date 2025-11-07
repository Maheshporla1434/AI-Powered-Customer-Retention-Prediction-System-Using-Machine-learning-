
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sys
from log_code import Logger
logger=Logger.get_logs('corelation')
from scipy.stats import ttest_ind,pearsonr,spearmanr
def co_relation(X_train_num,y_train):
    try:
        # log.info(y)
        # log.info(y.isnull().sum())
        # y = y.map({'Yes':1, 'No':0}).astype(int)
        features_significant = []
        for i in X_train_num.columns:
            r, p = spearmanr(X_train_num[i], y_train)
            logger.info(f'{i}----->{r}')
            if p < 0.05:
                features_significant.append(i)
        logger.info(f'spearman rank correlation:{features_significant}')
        features_significant = []
        for i in X_train_num.columns:
            r, p = pearsonr(X_train_num[i], y_train)
            logger.info(f'{i}----->{r}')
            if p < 0.05:
                features_significant.append(i)
        logger.info(f'pearson rank {features_significant}')


        return  X_train_num,y_train
    except Exception as e:
        exc_type, exc_msg, exc_line = sys.exc_info()
        logger.error(f'{exc_type} at {exc_line.tb_lineno} as {exc_msg}')
