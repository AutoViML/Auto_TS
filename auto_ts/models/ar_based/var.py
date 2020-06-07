import numpy as np  # type: ignore
import pandas as pd # type: ignore
import itertools
import operator
import copy
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
# This gives an error when running from a python script. 
# Maybe, this should be set in the jupyter notebook directly.
# get_ipython().magic('matplotlib inline')
sns.set(style="white", color_codes=True)
# imported VARMAX from statsmodels pkg
from statsmodels.tsa.statespace.varmax import VARMAX # type: ignore
# helper functions
from ...utils import print_dynamic_rmse
from ...models.ar_based.param_finder import find_lowest_pq


def build_var_model(df, criteria, forecast_period=2, p_max=3, q_max=3, verbose=0):
    """
    This builds a VAR model given a multivariate time series data frame with time as the Index.
    Note that the input "y_train" can be a data frame with one column or multiple cols or a
    multivariate array. However, the first column must be the target variable. The others are added.
    You must include only Time Series data in it. DO NOT include "Non-Stationary" or "Trendy" data.
    Make sure your Time Series is "Stationary" before you send it in!! If not, this will give spurious
    results. Since it automatically builds a VAR model, you need to give it a Criteria to optimize on.
    You can give it any of the following metrics as criteria: AIC, BIC, Deviance, Log-likelihood.
    You can give the highest order values for p and q. Default is set to 3 for both.
    """
    df = df[:]
    #### dmax here means the column number of the data frame: it serves as a placeholder for columns
    dmax = df.shape[1]
    ###############################################################################################
    cols = df.columns.tolist()
    ts_train = df[:-forecast_period]
    ts_test = df[-forecast_period:]
    if verbose == 1:
        print('Data Set split into train %s and test %s for Cross Validation Purposes'
              % (ts_train.shape, ts_test.shape))
    # It is assumed that the first column of the dataframe is the target variable ####
    ### make sure that is the case before doing this program ####################
    i = 1
    results_dict = {}
    for d_val in range(1, dmax):
        y_train = ts_train.iloc[:, [0, d_val]]
        print('\nAdditional Variable in VAR model = %s' % cols[d_val])
        info_criteria = pd.DataFrame(index=['AR{}'.format(i) for i in range(0, p_max+1)],
                                     columns=['MA{}'.format(i) for i in range(0, q_max+1)])
        for p_val, q_val in itertools.product(range(0, p_max+1), range(0, q_max+1)):
            if p_val == 0 and q_val == 0:
                info_criteria.loc['AR{}'.format(p_val), 'MA{}'.format(q_val)] = np.nan
                print(' Iteration %d completed' % i)
                i += 1
            else:
                try:
                    model = VARMAX(y_train, order=(p_val, q_val), trend='c')
                    model = model.fit(max_iter=1000, displ=False)
                    info_criteria.loc['AR{}'.format(p_val), 'MA{}'.format(q_val)] = eval('model.' + criteria)
                    print(' Iteration %d completed' % i)
                    i += 1
                except:
                    i += 1
                    print(' Iteration %d completed' % i)
        info_criteria = info_criteria[info_criteria.columns].astype(float)
        interim_d = copy.deepcopy(d_val)
        interim_p, interim_q, interim_bic = find_lowest_pq(info_criteria)
        if verbose == 1:
            fig, ax = plt.subplots(figsize=(20, 10))
            ax = sns.heatmap(info_criteria,
                             mask=info_criteria.isnull(),
                             ax=ax,
                             annot=True,
                             fmt='.0f'
                             )
            ax.set_title(criteria)
        results_dict[str(interim_p) + ' ' + str(interim_d) + ' ' + str(interim_q)] = interim_bic
    best_bic = min(results_dict.items(), key=operator.itemgetter(1))[1]
    best_pdq = min(results_dict.items(), key=operator.itemgetter(1))[0]
    best_p = int(best_pdq.split(' ')[0])
    best_d = int(best_pdq.split(' ')[1])
    best_q = int(best_pdq.split(' ')[2])
    print('Best variable selected for VAR: %s' % ts_train.columns.tolist()[best_d])
    y_train = ts_train.iloc[:, [0, best_d]]
    bestmodel = VARMAX(y_train, order=(best_p, best_q), trend='c')
    bestmodel = bestmodel.fit()
    if verbose == 1:
        bestmodel.plot_diagnostics(figsize=(16, 12))
        ax = bestmodel.impulse_responses(12, orthogonalized=True).plot(figsize=(12, 4))
        ax.set(xlabel='Time Steps', title='Impulse Response Functions')
    res2 = bestmodel.get_forecast(forecast_period)
    res2_df = res2.summary_frame()
    rmse, norm_rmse = print_dynamic_rmse(ts_test.iloc[:,0], res2_df['mean'].values, ts_train.iloc[:,0])
    return bestmodel, res2_df, rmse, norm_rmse
