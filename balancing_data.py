import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sys
from log_code import Logger
logger=Logger.get_logs('balancing_data')
from imblearn.over_sampling import SMOTE
def balanced_data(training_data,y_train):
    try:
        logger.info('----------------Before Balancing------------------------')
        logger.info(
            f'Total row for Good category in training data {training_data.shape[0]} was : {sum(y_train == 1)}')
        logger.info(
            f'Total row for Bad category in training data {training_data.shape[0]} was : {sum(y_train == 0)}')
        logger.info(f'---------------After Balancing-------------------------')
        sm = SMOTE(random_state=42)
        training_data_res, y_train_res = sm.fit_resample(training_data, y_train)
        logger.info(
            f'Total row for Good category in training data {training_data_res.shape[0]} was : {sum(y_train_res == 1)}')
        logger.info(
            f'Total row for Bad category in training data {training_data_res.shape[0]} was : {sum(y_train_res == 0)}')


        return training_data_res,y_train_res
    except Exception as e:
        er_ty, er_msg, er_lin = sys.exc_info()
        logger.info(f'Issue is : {er_lin.tb_lineno} : due to : {er_msg}')