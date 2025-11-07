import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import sys
from feature_engine.outliers import Winsorizer
from log_code import Logger
from scipy.stats import ttest_ind,pearsonr
from visual import Visual
logger=Logger.get_logs('filters')
from sklearn.feature_selection import SelectKBest, chi2,mutual_info_classif,f_classif
def chi_square_test(X_train_cat,X_train_num,X_test_cat,X_test_num,y_train,y_test ,p_threshold=0.05):
    try:

        '''
        CHI SQUARE TEST
        '''
        logger.info(X_train_cat.isnull().sum())
        selector =SelectKBest(score_func=chi2, k=10)
        X_new = selector.fit_transform(X_train_cat, y_train)
        logger.info(f'X_new.shape: {X_new}')
        chi2_score = pd.DataFrame(
            {
                'Features': X_train_cat.columns,
                'Chi_score': selector.scores_,
                'p values' :selector.pvalues_
            }
        ).sort_values(by='Chi_score', ascending=False)
        logger.info(f'Chi score data frame :\n{chi2_score}')
        #
        #
        chi2_score = chi2_score[chi2_score['Features' ]!= 'sim']
        remove_features = chi2_score[chi2_score['p values'] > 0.05]['Features']
        df_filtered = X_train_cat.drop(columns=remove_features)
        X_train_cat1 =df_filtered
        df_filtered1 = X_test_cat.drop(columns=remove_features)
        X_test_cat1 =df_filtered1
        logger.info(f'Removed features: {remove_features}')
        logger.info(f'{X_train_cat1.head()}')

        # '''
        # Mutual Information
        # '''
        #
        # mi_selector = SelectKBest(score_func=mutual_info_classif, k='all')
        # mi = mi_selector.fit(self.X_train_cat, self.y_train)
        # mi_scores = pd.DataFrame({
        #     'Feature': self.X_train_cat.columns,
        #     'MI_Score': mi_selector.scores_
        # }).sort_values(by='MI_Score', ascending=False)
        # remove_features = mi_scores[mi_scores['MI_Score'] > 0.05]['Feature']
        # df_filtered1 = self.X_train_cat.drop(columns=remove_features)
        # logger.info(f'Removed features: {remove_features}')

        X_train_cat1['sim' ] =X_train_cat['sim']
        X_test_cat1['sim' ] =X_test_cat['sim']

        '''
        F-TEST
        '''
        anova_selector = SelectKBest(score_func=f_classif, k='all')
        anova_selector.fit(X_train_num, y_train)
        anova_df = pd.DataFrame({
            'Feature': X_train_num.columns,
            'F_Score': anova_selector.scores_,
            'p_value': anova_selector.pvalues_
        }).sort_values(by='F_Score', ascending=False)

        logger.info(f"ANOVA Results:\n{anova_df}")
        remove_features = anova_df[anova_df['p_value'] > 0.05]['Feature']
        df_filtered2 = X_train_num.drop(columns=remove_features)
        df_filtered3 = X_test_num.drop(columns=remove_features)
        X_train_num = df_filtered2
        X_test_num = df_filtered3


        logger.info(f'Removed features: {X_train_num.head()}')

        logger.info(f'{X_train_num.head()}')
        logger.info(f'{y_train}')
        selected = []
        for i in X_train_num.columns:
            g1 = X_train_num[y_train == 0][i]
            g2 = X_train_num[y_train == 1][i]
            if g1.empty or g2.empty:
                logger.warning(f"{i}: empty groups, check y alignment.")
                continue

            t, p = ttest_ind(g1, g2, nan_policy='omit')
            logger.info(f"T={t:.3f}, P={p:.5f}")
            if p < 0.05:
                selected.append(i)

            logger.info(f"Selected Features: {selected}")

            return X_train_cat1,X_test_cat1,X_train_num ,X_test_num




    except Exception:
        exc_type, exc_msg, exc_line = sys.exc_info()
        print(exc_line, exc_type, exc_msg)
