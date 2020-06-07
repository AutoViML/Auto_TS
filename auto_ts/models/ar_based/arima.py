import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import itertools
import operator
import copy
import seaborn as sns  # type: ignore
import matplotlib.pyplot as plt
# This gives an error when running from a python script. 
# Maybe, this should be set in the jupyter notebook directly.
# get_ipython().magic('matplotlib inline')
sns.set(style="white", color_codes=True)
# imported ARIMA from statsmodels pkg
from statsmodels.tsa.arima_model import ARIMA # type: ignore
# helper functions
from ...utils import print_static_rmse, print_dynamic_rmse
from ...models.ar_based.param_finder import find_lowest_pq
import pdb

def build_arima_model(ts_df, metric='aic', p_max=3, d_max=1, q_max=3,
                      forecast_period=2, method='mle', verbose=0):
    """
    This builds a Non Seasonal ARIMA model given a Univariate time series dataframe with time
    as the Index, ts_df can be a dataframe with one column only or a single array. Dont send
    Multiple Columns!!! Include only that variable that is a Time Series. DO NOT include
    Non-Stationary data. Make sure your Time Series is "Stationary"!! If not, this
    will give spurious results, since it automatically builds a Non-Seasonal model,
    you need not give it a Seasonal True/False flag.
    "metric": You can give it any of the following metrics as criteria: AIC, BIC, Deviance,
    Log-likelihood. Optionally, you can give it a fit method as one of the following:
    {'css-mle','mle','css'}
    """
    p_min = 0
    d_min = 0
    q_min = 0
    # Initialize a DataFrame to store the results
    iteration = 0
    results_dict = {}
    ################################################################################
    ####### YOU MUST Absolutely set this parameter correctly as "levels". If not,
    ####  YOU WILL GET DIFFERENCED PREDICTIONS WHICH ARE FIENDISHLY DIFFICULT TO UNDO.
    #### If you set this to levels, then you can do any order of differencing and
    ####  ARIMA will give you predictions in the same level as orignal values.
    ################################################################################
    pred_type = 'levels'
    #########################################################################
    ts_train = ts_df[:-forecast_period]
    ts_test = ts_df[-forecast_period:]
    if verbose == 1:
        print('Data Set split into train %s and test %s for Cross Validation Purposes'
              % (ts_train.shape, ts_test.shape))
    #########################################################################
    if ts_train.dtype == 'int64':
        ts_train = ts_train.astype(float)
    for d_val in range(d_min, d_max+1):
        print('\nDifferencing = %d' % d_val)
        results_bic = pd.DataFrame(index=['AR{}'.format(i) for i in range(p_min, p_max+1)],
                                   columns=['MA{}'.format(i) for i in range(q_min, q_max+1)])
        for p_val, q_val in itertools.product(range(p_min, p_max+1), range(q_min, q_max+1)):
            if p_val == 0 and d_val == 0 and q_val == 0:
                results_bic.loc['AR{}'.format(p_val), 'MA{}'.format(q_val)] = np.nan
                continue
            else:
                try:
                    model = ARIMA(ts_train, order=(p_val, d_val, q_val))
                    results = model.fit(transparams=False, method=method)
                    results_bic.loc['AR{}'.format(p_val), 'MA{}'.format(q_val)] = eval('results.' + metric)
                    if iteration % 10 == 0:
                        print(' Iteration %d completed...' % iteration)
                    iteration += 1
                    if iteration >= 100:
                        print('    Ending Iterations at %d' % iteration)
                        break
                except:
                    iteration += 1
                    continue
        results_bic = results_bic[results_bic.columns].astype(float)
        interim_d = copy.deepcopy(d_val)
        interim_p, interim_q, interim_bic = find_lowest_pq(results_bic)
        if verbose == 1:
            _, ax = plt.subplots(figsize=(20, 10))
            ax = sns.heatmap(results_bic,
                             mask=results_bic.isnull(),
                             ax=ax,
                             annot=True,
                             fmt='.0f')
            ax.set_title(metric)
        results_dict[str(interim_p) + ' ' + str(interim_d) + ' ' + str(interim_q)] = interim_bic
    best_bic = min(results_dict.items(), key=operator.itemgetter(1))[1]
    best_pdq = min(results_dict.items(), key=operator.itemgetter(1))[0]
    best_p = int(best_pdq.split(' ')[0])
    best_d = int(best_pdq.split(' ')[1])
    best_q = int(best_pdq.split(' ')[2])
    print('\nBest model is: Non Seasonal ARIMA(%d,%d,%d), %s = %0.3f' % (best_p, best_d, best_q,metric, best_bic))
    bestmodel = ARIMA(ts_train, order=(best_p, best_d, best_q))
    print('####    Fitting best model for full data set now. Will take time... ######')
    try:
        results = bestmodel.fit(transparams=True, method=method)
    except:
        results = bestmodel.fit(transparams=False, method=method)
    ### this is needed for static forecasts ####################
    y_truth = ts_train[:]
    y_forecasted = results.predict(typ='levels')
    concatenated = pd.concat([y_truth, y_forecasted], axis=1, keys=['original', 'predicted'])
    if best_d == 0:
        #### Do this for ARIMA only ######
        ###  If there is no differencing DO NOT use predict_type since it will give an error = do not use "linear".
        print('Static Forecasts:')
        print_static_rmse(concatenated['original'].values, concatenated['predicted'].values, best_d)
        start_date = ts_df.index[-forecast_period]
        end_date = ts_df.index[-1]
        pred_dynamic = results.predict(start=start_date, end=end_date, dynamic=True)
        if verbose == 1:
            ax = concatenated[['original', 'predicted']][best_d:].plot()
            pred_dynamic.plot(label='Dynamic Forecast', ax=ax, figsize=(15, 5))
            print('Dynamic %d-period Forecasts:' % (forecast_period,))
            plt.legend()
            plt.show(block=False)
    else:
        #### Do this for ARIMA only ######
        ####  If there is differencing, you must use "levels" as the predict type to get original levels as actuals
        pred_type = 'levels'
        print('Static Forecasts:')
        print_static_rmse(y_truth[best_d:], y_forecasted)
        ########### Dynamic One Step Ahead Forecast ###########################
        ### Dynamic Forecasts are a better representation of true predictive power
        ## since they only use information from the time series up to a certain point,
        ## and after that, forecasts are generated using values from previous forecasted
        ## time points.
        #################################################################################
        start_date = ts_df.index[-forecast_period]
        end_date = ts_df.index[-1]
        pred_dynamic = results.predict(typ=pred_type, start=start_date, end=end_date, dynamic=True)
        try:
            pred_dynamic[pd.to_datetime((pred_dynamic.index-best_d).values[0])] = \
                                     y_truth[pd.to_datetime((pred_dynamic.index-best_d).values[0])]
        except:
            print('Dynamic predictions erroring but continuing...')
        pred_dynamic.sort_index(inplace=True)
        print('\nDynamic %d-period Forecasts:' % forecast_period)
        if verbose == 1:
            ax = concatenated.plot()
            pred_dynamic.plot(label='Dynamic Forecast', ax=ax, figsize=(15, 5))
            ax.set_xlabel('Date')
            ax.set_ylabel('Values')
            plt.legend()
            plt.show(block=False)
    if verbose == 1:
        try:
            results.plot_diagnostics(figsize=(16, 12))
        except:
            pass
    print(results.summary())
    res_frame = pd.DataFrame([results.forecast(forecast_period)[0], results.forecast(forecast_period)[1],
                                               results.forecast(forecast_period)[2]],
                                               index=['mean','mean_se','mean_ci'],
                                               columns=['Forecast_' + str(x) for x
                                               in range(1, forecast_period+1)]).T
    res_frame['mean_ci_lower'] = res_frame['mean_ci'].map(lambda x: x[0])
    res_frame['mean_ci_upper'] = res_frame['mean_ci'].map(lambda x: x[1])
    res_frame.drop('mean_ci', axis=1, inplace=True)
    if verbose == 1:
        print('Model Forecast(s):\n', res_frame)
    rmse, norm_rmse = print_dynamic_rmse(ts_test, pred_dynamic, ts_train)
    return results, res_frame, rmse, norm_rmse


def predicted_diffs_restored_ARIMA(actuals, predicted, periods=1):
    """
    This utility is needed only we dont set typ="levels" in arima.fit() method.
    Hence this utility caters only to ARIMA models in a few cases. Don't need it.
    """
    if periods == 0:
        restored = predicted.copy()
        restored.sort_index(inplace=True)
        restored[0] = actuals[0]
    else:
        restored = actuals.copy()
        restored.iloc[periods:] = predicted[periods:]
        restored = restored[(periods-1):].cumsum()
    res = pd.concat([actuals, predicted, restored], axis=1)
    res.columns = ['original', 'pred_as_diffs', 'predicted']
    print_static_rmse(res['original'].values, res['predicted'].values, periods-1)
    return res[['original', 'predicted']]
