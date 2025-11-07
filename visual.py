import matplotlib.pyplot as plt
from log_code import Logger
logger = Logger.get_logs('visual')
import matplotlib.pyplot as plt
from scipy import stats
import sys
import seaborn as sns

class Visual:
    def plot_kde_comparison(df, transform_suffix, color, label):
        try:
            logger.info('visual stated')
            for col in df.columns:
                if col.endswith(transform_suffix):
                    logger.info(col)
                    plt.figure(figsize=(10, 4))
                    # KDE and boxplot in one figure (2 subplots)
                    plt.subplot(1, 3, 1)
                    df[col].plot(kind='kde', color=color, label=label)
                    plt.title(f'{label} - {col}')
                    logger.info('kde-st')
                    plt.legend()

                    plt.subplot(1, 3, 2)
                    plt.boxplot(df[col].dropna(), labels=[label])
                    logger.info('box-st')
                    plt.title(f'Boxplot - {col}')

                    plt.subplot(1, 3, 3)
                    stats.probplot(df[col].dropna(), dist='norm', plot=plt)
                    logger.info('prob-st')
                    plt.title('Probplot')
                    plt.tight_layout()
                    # plt.savefig(f'./image/{label}-{transform_suffix}-{col}.jpeg')
                    plt.show()
                    logger.info(f'{col} graph {label}completed')


        except Exception:
            exc_type, exc_msg, exc_tb = sys.exc_info()
            logger.error(f"{exc_type} at line {exc_tb.tb_lineno}: {exc_msg}")

    def fun(data_cut, var):
        try:
            logger.info("Visual2 has started........")
            plt.subplot(1, 3, 1)
            sns.kdeplot(data_cut[var].dropna(), fill=True, color='skyblue')
            plt.title(f'KDE Plot - {var}')

            # Probability Plot
            plt.subplot(1, 3, 2)
            stats.probplot(data_cut[var].dropna(), dist='norm', plot=plt)
            plt.title(f'Probability Plot - {var}')

            # Boxplot
            plt.subplot(1, 3, 3)
            sns.boxplot(y=data_cut[var], color='lightgreen')
            plt.title(f'Boxplot - {var}')
            plt.ylabel("Values")
            plt.show()

        except Exception:
            exc_type, exc_msg, exc_tb = sys.exc_info()
            logger.error(f"{exc_type} at line {exc_tb.tb_lineno}: {exc_msg}")