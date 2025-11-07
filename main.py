import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sys
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import logging
import pickle
from balancing_data import balanced_data
from log_code import Logger
logger=Logger.get_logs('main')
from parameter_check import check
from algorithms import common
from missing_values import MISSINGVALUES
from imblearn.over_sampling import SMOTE
from filter_methods import chi_square_test
from visual import Visual
from merge_data import merge_data1
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from corelation import co_relation
from encoding import encoding_cat
from outlier_handling import OUTLIER
from variable_transformation import quantile_transform_check
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
class CHURN:
    try:
        def __init__(self,path):
            try:
                self.df=pd.read_csv(path)
                logger.info(f"Top five rows from dataset:{self.df.head()}")
                logger.info(f' Total Columns: {self.df.columns}')
                self.df['TotalCharges'] = pd.to_numeric(self.df['TotalCharges'], errors='coerce')
                self.X = self.df.drop(['customerID', 'Churn'], axis=1)
                self.y = self.df['Churn']

                self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2,
                                                                                        random_state=42)

                logger.info(f'x_test :{self.X_test.columns}')
                logger.info(f'x_train:{self.X_train.columns}')

                logger.info(f'x_test shape:{self.X_test.shape}')
                logger.info(f'x_train shape:{self.X_train.shape}')
                logger.info(f'y_train shape:{self.y_train.shape}')
                logger.info(f'y_test shape:{self.y_test.shape}')

                self.total_dummy_charges_Xtrain = pd.DataFrame()
                self.total_dummy_charges_Xtest = pd.DataFrame()
                self.total_dummy_charges_Xtrain['Total_charges'] = self.X_train['TotalCharges']
                self.total_dummy_charges_Xtest['Total_charges'] = self.X_test['TotalCharges']
                logger.info(f' checking null values in x_train {self.total_dummy_charges_Xtrain.isnull().sum()}')
                logger.info(f' checking null values in x_test {self.total_dummy_charges_Xtest.isnull().sum()}')

            except Exception:
                exc_type, exc_msg, exc_tb = sys.exc_info()
                logger.error(f"{exc_type} at line {exc_tb.tb_lineno}:{exc_msg}")


        def MISSING_VALUES(self):
            try:

                self.dummy_x_train,self.dummy_x_test,self.X_train,self.X_test=MISSINGVALUES(self.total_dummy_charges_Xtrain,self.total_dummy_charges_Xtest,self.X_train,self.X_test)
                logger.info(f'dummy_x_train :{self.dummy_x_train.isnull().sum()}')
                logger.info(f'dummy_x_test :{self.dummy_x_test.isnull().sum()}')

            except Exception:
                exc_type, exc_msg, exc_tb = sys.exc_info()
                logger.error(f"{exc_type} at line {exc_tb.tb_lineno}:{exc_msg}")


        def VARIABLE_TRANSFORMATION(self):
            '''
            variable transformation using quantile_transform_check

            '''

            try:
                logger.info(f'started : {self.dummy_x_train.columns}')
                self.X_train,self.X_test=quantile_transform_check(self.dummy_x_train,self.dummy_x_test,self.X_train,self.X_test)

            except Exception:
                exc_type, exc_msg, exc_tb = sys.exc_info()
                logger.error(f"{exc_type} at line {exc_tb.tb_lineno}:{exc_msg}")



        def outliers(self):
            '''
            variable transformation using quantile_transform_check

            '''
            try:
                self.X_train,self.X_test=OUTLIER(self.X_train,self.X_test)

            except Exception:
                exc_type, exc_msg, exc_tb = sys.exc_info()
                logger.error(f"{exc_type} at line {exc_tb.tb_lineno}:{exc_msg}")




        def dividing(self):
            try:
                self.X_train_cat = self.X_train.select_dtypes(include=['object', 'category'])
                self.X_train_num = self.X_train.select_dtypes(include=['number'])
                self.X_test_cat = self.X_test.select_dtypes(include=['object', 'category'])
                self.X_test_num = self.X_test.select_dtypes(include=['number'])
                logger.info(f'x_test_cat :{self.X_test_cat.columns}')
                logger.info(f'x_train_cat:{self.X_train_cat.columns}')
                logger.info(f'x_test_num :{self.X_test_num.columns}')
                logger.info(f'x_train_num:{self.X_train_num.columns}')

            except Exception:
                exc_type, exc_msg, exc_tb = sys.exc_info()
                logger.error(f"{exc_type} at line {exc_tb.tb_lineno}:{exc_msg}")


        def encoding1(self):
            '''
            variable transformation using quantile_transform_check

            '''
            try:
                self.X_train_cat,self.X_train_num,self.X_test_cat,self.X_test_num,self.y_train,self.y_test=encoding_cat(self.X_train_cat,self.X_train_num,self.X_test_cat,self.X_test_num,self.y_train,self.y_test)

            except Exception:
                exc_type, exc_msg, exc_tb = sys.exc_info()
                logger.error(f"{exc_type} at line {exc_tb.tb_lineno}:{exc_msg}")

        def FILTER(self):
            '''
            variable transformation using quantile_transform_check

            '''
            try:
                self.X_train_cat1,self.X_test_cat1,self.X_train_num ,self.X_test_num =chi_square_test(self.X_train_cat,self.X_train_num,self.X_test_cat,self.X_test_num,self.y_train,self.y_test)

            except Exception:
                exc_type, exc_msg, exc_tb = sys.exc_info()
                logger.error(f"{exc_type} at line {exc_tb.tb_lineno}:{exc_msg}")

        def correlation(self):
            '''
            Hypothesis

            '''
            try:
                self.X_train_num ,self.y_train =co_relation(self.X_train_num,self.y_train)

            except Exception:
                exc_type, exc_msg, exc_tb = sys.exc_info()
                logger.error(f"{exc_type} at line {exc_tb.tb_lineno}:{exc_msg}")


        def MERGINGDATA(self):
            '''
            Hypothesis

            '''
            try:
                self.training_data,self.testing_data =merge_data1(self.X_train_cat1,self.X_test_cat1,self.X_train_num ,self.X_test_num)

            except Exception:
                exc_type, exc_msg, exc_tb = sys.exc_info()
                logger.error(f"{exc_type} at line {exc_tb.tb_lineno}:{exc_msg}")




        def BALANCING_DATA(self):
            '''
            Hypothesis

            '''
            try:
                self.training_data_res,self.y_train_res =balanced_data(self.training_data,self.y_train)
                logger.info(f'111111111{self.training_data_res.columns}')

            except Exception:
                exc_type, exc_msg, exc_tb = sys.exc_info()
                logger.error(f"{exc_type} at line {exc_tb.tb_lineno}:{exc_msg}")

        def feature_scaling(self):
            try:
                # logger.info('---------Before scaling standardscalar-------')
                # logger.info(f'{self.training_data_res.head(4)}')
                # sc = StandardScaler()
                # sc.fit(self.training_data_res)
                # self.training_data_res_t = sc.transform(self.training_data_res)
                # self.testing_data_t = sc.transform(self.testing_data)
                # # with open('standard_scalar.pkl', 'wb') as t:
                # #     pickle.dump(sc, t)
                # logger.info('----------After scaling using standard scalar--------')
                # logger.info(f'{self.training_data_res_t}')
                backup = self.training_data_res['JoinYear'].copy()
                logger.info(f'JoinYear : {backup}')
                backup1 = self.testing_data['JoinYear'].copy()
                self.training_data_res = self.training_data_res.drop(['JoinYear'], axis=1)
                self.testing_data = self.testing_data.drop(['JoinYear'],axis = 1)
                logger.info('---------Before scaling standardscalar-------')
                logger.info(f'{self.training_data_res.head(4)}')
                sc = StandardScaler()

                scale_cols = ['MonthlyCharges', 'TotalCharges']
                self.ms = StandardScaler()
                self.ms.fit(self.training_data_res[scale_cols])

                scaled_train = pd.DataFrame(
                    self.ms.transform(self.training_data_res[scale_cols]),
                    columns=scale_cols, index=self.training_data_res.index)
                scaled_test = pd.DataFrame(
                    self.ms.transform(self.testing_data[scale_cols]),
                    columns=scale_cols, index=self.testing_data.index)

                other_train = self.training_data_res.drop(scale_cols, axis=1, errors='ignore')
                other_test = self.testing_data.drop(scale_cols, axis=1, errors='ignore')

                self.training_data_t = pd.concat([other_train, scaled_train], axis=1)
                self.testing_data_t = pd.concat([other_test, scaled_test], axis=1)
                self.training_data_t['JoinYear'] = backup
                self.testing_data_t['JoinYear'] = backup1
                with open('standard_scalar.pkl', 'wb') as t:
                    pickle.dump(self.ms, t)

                logger.info('----------After scaling using standard scalar--------')
                logger.info(self.training_data_t.head(4))
                logger.info(f'successfully executed')
                # Save the final training columns (after encoding)


            except Exception as e:
                er_ty, er_msg, er_lin = sys.exc_info()
                logger.info(f'Issue is : {er_lin.tb_lineno} : dueto :{er_msg}')

        def train_models(self):
            try:
                common(self.training_data_t, self.y_train_res, self.testing_data_t, self.y_test)
            except Exception as e:
                er_ty, er_msg, er_lin = sys.exc_info()
                logger.info(f'Issue is : {er_lin.tb_lineno} : dueto :{er_msg}')


        def paramters(self):
            try:
                # self.train_ind=self.scaled_df.head(200)
                # self.train_dep=self.y_train_res[:200]
                # check(self.train_ind, self.train_dep)
                logger.info(f'_Finalized Model')
                self.reg1 = LogisticRegression(C=1.0, class_weight=None, l1_ratio=None, max_iter=100,
                                               multi_class='auto', n_jobs=None, penalty='l2', solver='lbfgs')
                self.reg1.fit(self.training_data_t, self.y_train_res)
                logger.info(f'Train accuracy:{accuracy_score(self.y_train_res, self.reg1.predict(self.training_data_t))}')
                logger.info(f'Test accuracy:{accuracy_score(self.y_test, self.reg1.predict(self.testing_data_t))}')
                logger.info(f'=====Model Saving======')
                with open('churn_model.pkl', 'wb') as f:
                    pickle.dump(self.reg1, f)

                logger.info(f'{self.training_data_t.columns}')



            except Exception as e:
                er_ty, er_msg, er_lin = sys.exc_info()
                logger.info(f'Issue is : {er_lin.tb_lineno} : dueto :{er_msg}')

    except Exception:
        exc_type, exc_msg, exc_tb = sys.exc_info()
        logger.error(f"{exc_type} at line {exc_tb.tb_lineno}:{exc_msg}")



if __name__ == '__main__':
    path='C:\\Users\\Mahesh Porla\\Downloads\\internship\\churn_prediction.csv'
    obj=CHURN(path)
    obj.MISSING_VALUES()
    obj.VARIABLE_TRANSFORMATION()
    obj.outliers()
    obj.dividing()
    obj.encoding1()
    obj.FILTER()
    obj.correlation()
    obj.MERGINGDATA()
    obj.BALANCING_DATA()
    obj.feature_scaling()
    obj.train_models()
    obj.paramters()


