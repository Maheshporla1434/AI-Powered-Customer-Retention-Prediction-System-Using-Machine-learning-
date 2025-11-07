
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sys
from log_code import Logger
logger=Logger.get_logs('variable')
from visual import Visual
from sklearn.preprocessing import quantile_transform


def quantile_transform_check(dummy_x_train,dummy_x_test,X_train,X_test ):
    try:
        logger.info(f'dummy_x_train :{dummy_x_train.isnull().sum()}')
        logger.info(f'dummy_x_test :{dummy_x_test.isnull().sum()}')
        transformed = quantile_transform(dummy_x_train, output_distribution='normal', random_state=42)
        df_qt = pd.DataFrame(transformed, columns=[col + '_qt' for col in dummy_x_train.columns],
                             index=dummy_x_train.index)
        transformed1 = quantile_transform(dummy_x_test, output_distribution='normal', random_state=42)
        df_pt = pd.DataFrame(transformed1, columns=[col + '_qt' for col in dummy_x_train.columns],
                             index=dummy_x_test.index)
        logger.info("Quantile transformation completed successfully.")
        logger.info(df_qt.isnull().sum())
        logger.info(df_qt.head(10))
        logger.info(df_qt.columns)
        # Visual.plot_kde_comparison(df_qt,'_qt',"r",'quantile transform')
        # Visual.plot_kde_comparison(df_pt, '_qt', "r", 'quantile transform')
        logger.info(f'check{df_qt}')
        X_train['MonthlyCharges_qt'] = df_qt['MonthlyCharges_qt']
        X_train['TotalCharges_qt'] = df_qt['TotalCharges_qt']
        X_test['MonthlyCharges_qt'] = df_pt['MonthlyCharges_qt']
        X_test['TotalCharges_qt'] = df_pt['TotalCharges_qt']
        logger.info(X_train.columns)


        return X_train,X_test





    except Exception:
        exc_type, exc_msg, exc_tb = sys.exc_info()
        logger.error(f"{exc_type} at line {exc_tb.tb_lineno}:{exc_msg}")