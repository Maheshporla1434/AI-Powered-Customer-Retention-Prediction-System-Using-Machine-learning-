import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sys
from feature_engine.outliers import Winsorizer
from log_code import Logger
from visual import Visual
logger=Logger.get_logs('outlier')
def OUTLIER(X_train,X_test):
    try:
        num_cols = ['MonthlyCharges_qt', 'TotalCharges_qt']
        df_num = X_train[num_cols].copy()
        df_num1 = X_test[num_cols].copy()
        # logger.info(f'Check the error: {self.X_test[num_cols]}')
        # logger.info(f'Checking the shape before trim:{df_num.shape}')
        # logger.info(df_num.isnull().sum())
        # logger.info(df_num.columns)
        winsor = Winsorizer(
            capping_method='gaussian',  # method to calculate caps
            tail='both',  # cap both tails (upper & lower)
            fold=2.5,  # equivalent to 1.5 * IQR

        )
        logger.info(df_num1.info())
        df_winsor = winsor.fit_transform(df_num)
        logger.info(df_num1)
        df_winsor1 = winsor.fit_transform(df_num1)


        for i in df_winsor.columns:
            Visual.fun(df_winsor ,i)


        logger.info(f'Checking the shape After trim:{df_winsor.shape}')

        logger.info(X_train.columns)
        logger.info(X_test.columns)
        X_train['MonthlyCharges'] = df_winsor['MonthlyCharges_qt']
        X_test['MonthlyCharges'] = df_winsor1['MonthlyCharges_qt']
        X_train['TotalCharges'] = df_winsor['TotalCharges_qt']
        X_test['TotalCharges'] = df_winsor1['TotalCharges_qt']
        X_train = X_train.drop(['MonthlyCharges_qt', 'TotalCharges_qt'], axis=1)
        X_test = X_test.drop(['MonthlyCharges_qt', 'TotalCharges_qt'] ,axis = 1)


        return X_train,X_test
    except Exception:
        exc_type, exc_msg, exc_tb = sys.exc_info()
        logger.error(f"{exc_type} at line {exc_tb.tb_lineno}:{exc_msg}")

