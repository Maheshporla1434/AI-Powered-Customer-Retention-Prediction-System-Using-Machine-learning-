import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
from feature_engine.imputation import RandomSampleImputer
from log_code import Logger
logger=Logger.get_logs('missing')

def MISSINGVALUES(total_dummy_charges_Xtrain,total_dummy_charges_Xtest,X_train,X_test):
    try:
        logger.info(f'before updating missing values x train{X_train.isnull().sum()}')
        logger.info(f'before updating missing values x train{X_test.isnull().sum()}')
        logger.info('MISSING VALUES started')
        rsi = RandomSampleImputer(random_state=42)
        b = rsi.fit_transform(X_train)
        bb =rsi.fit_transform(X_test)
        plt.figure(figsize=(8, 4))
        plt.subplot(2, 2, 1)
        total_dummy_charges_Xtrain['Total_charges'].hist(bins=30, color='red',
                                                              label=f"Before-{total_dummy_charges_Xtrain['Total_charges'].std()}")
        plt.title("Before Imputation x_train (RandomSampleImputer) ")
        plt.legend()
        plt.subplot(2, 2, 2)
        b['TotalCharges'].hist(bins=30, color='blue', label=f"after-{b['TotalCharges'].std()}")
        plt.title("After Imputation x_train")
        plt.legend()
        plt.subplot(2, 2, 3)
        total_dummy_charges_Xtest['Total_charges'].hist(bins=30, color='red',
                                                             label=f"Before-{total_dummy_charges_Xtest['Total_charges'].std()}")
        plt.title("Before Imputation x_test")
        plt.legend()
        plt.subplot(2, 2, 4)
        bb['TotalCharges'].hist(bins=30, color='blue', label=f"after-{bb['TotalCharges'].std()}")
        plt.title("After Imputation x_test")
        plt.legend()
        plt.tight_layout()
        plt.show()
        X_train['TotalCharges'] = b['TotalCharges']
        X_test['TotalCharges'] = bb['TotalCharges']
        logger.info(f'after updating missing values x train{X_train.isnull().sum()}')
        logger.info(f'after updating missing values x train{X_test.isnull().sum()}')

        dummy_x_train = pd.DataFrame(
            {'MonthlyCharges': X_train['MonthlyCharges'], 'TotalCharges': X_train['TotalCharges']})
        dummy_x_test = pd.DataFrame(
            {'MonthlyCharges': X_test['MonthlyCharges'], 'TotalCharges': X_test['TotalCharges']})

        return dummy_x_train,dummy_x_test,X_train,X_test
    except Exception:
        exc_type, exc_msg, exc_tb = sys.exc_info()
        logger.error(f"{exc_type} at line {exc_tb.tb_lineno}:{exc_msg}")
