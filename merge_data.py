import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sys
from log_code import Logger
logger=Logger.get_logs('merge_data')



def merge_data1(X_train_cat1,X_test_cat1,X_train_num ,X_test_num):
    try:
        # reset index so that we can concat data perfectlly
        X_train_num.reset_index(drop=True, inplace=True)
        X_train_cat1.reset_index(drop=True, inplace=True)

        X_test_num.reset_index(drop=True, inplace=True)
        X_test_cat1.reset_index(drop=True, inplace=True)

        training_data = pd.concat([X_train_num, X_train_cat1], axis=1)
        testing_data = pd.concat([X_test_num, X_test_cat1], axis=1)

        logger.info(f'Training_data cat shape : {training_data.shape} -> {training_data.columns}')

        logger.info(f'Testing_data cat shape : {testing_data.shape} -> {testing_data.columns}')


        return  training_data,testing_data

    except Exception as e:
        er_ty, er_msg, er_lin = sys.exc_info()
        logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')