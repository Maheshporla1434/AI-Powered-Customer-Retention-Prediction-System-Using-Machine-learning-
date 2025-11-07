
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sys
from feature_engine.outliers import Winsorizer
from log_code import Logger
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
logger=Logger.get_logs('encoding')
def encoding_cat(X_train_cat,X_train_num,X_test_cat,X_test_num,y_train,y_test):
    try:
        cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
                'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                'StreamingMovies', 'PaperlessBilling', 'PaymentMethod']


        # one hot encoding
        oh = OneHotEncoder(categories='auto', drop='first', handle_unknown='ignore')
        oh.fit(X_train_cat[cols])
        logger.info(f'{oh.categories_}')
        logger.info(f'{oh.get_feature_names_out()}')
        res = oh.transform(X_train_cat[cols]).toarray()
        res1 = oh.transform(X_test_cat[cols]).toarray()
        f = pd.DataFrame(res, columns=oh.get_feature_names_out(), index=X_train_cat.index)
        f1 = pd.DataFrame(res1, columns=oh.get_feature_names_out(), index=X_test_cat.index)
        X_train_cat = pd.concat([X_train_cat.drop(columns=cols), f], axis=1)
        X_test_cat = pd.concat([X_test_cat.drop(columns=cols), f1], axis=1)
        logger.info(X_train_cat.isnull().sum())
        logger.info(X_test_cat.isnull().sum())

        # odinal encoding
        ob = OrdinalEncoder()
        ob.fit(X_train_cat[['Contract']])
        logger.info(f'{ob.categories_}')
        logger.info(f'column names:{ob.get_feature_names_out()}')
        res2 = ob.transform(X_train_cat[['Contract']])
        res2_test = ob.transform(X_test_cat[['Contract']])
        c_names = ob.get_feature_names_out()
        f2 = pd.DataFrame(res2, columns=c_names + ['_con'], index=X_train_cat.index)
        f2_test = pd.DataFrame(res2_test, columns=c_names + ['_con'], index=X_test_cat.index)
        X_train_cat = pd.concat([X_train_cat, f2], axis=1)
        X_test_cat = pd.concat([X_test_cat, f2_test], axis=1)
        X_train_cat = X_train_cat.drop(['Contract'], axis=1)
        X_test_cat = X_test_cat.drop(['Contract'], axis=1)
        logger.info(f'{X_train_cat.columns}')
        logger.info(f'{X_train_cat.sample(5)}')
        logger.info(f'{X_train_cat.isnull().sum()}')

        # label encoding
        logger.info(f'dependent:{y_train[:10]}')
        lb = LabelEncoder()
        lb.fit(y_train)
        y_train = lb.transform(y_train)
        y_test = lb.transform(y_test)
        logger.info(f'detailed:{lb.classes_}')
        logger.info(f'dependent:{y_train[:10]}')
        logger.info(f'y_train_data:{y_train.shape}')
        logger.info(f'y_test_data:{y_test.shape}')

        # 0 -> No
        # 1 -> Yes
        logger.info(f'Check null1 in before the drop {X_train_num["SeniorCitizen"]}')
        X_train_cat['SeniorCitizen'] = X_train_num['SeniorCitizen']
        X_test_cat['SeniorCitizen'] = X_test_num['SeniorCitizen']
        X_train_cat['sim'] = X_train_cat['sim'].map({'Jio' :0 ,'Airtel' :1 ,'Vi' :2 ,'BSNL' :3})
        X_test_cat['sim'] = X_test_cat['sim'].map({'Jio' :0 ,'Airtel' :1 ,'Vi' :2 ,'BSNL' :3})

        logger.info(X_train_cat)
        X_train_num = X_train_num.drop(['SeniorCitizen'], axis=1)
        X_test_num = X_test_num.drop(['SeniorCitizen'], axis=1)
        logger.info(f'check the null in the data frame:{X_train_cat.isnull().sum()}')

        return X_train_cat,X_train_num,X_test_cat,X_test_num,y_train,y_test
    except Exception as e:
        e_type, e_msg, e_linno = sys.exc_info()
        logger.info(f'Issue is:{e_linno.tb_lineno} due to {e_msg}')