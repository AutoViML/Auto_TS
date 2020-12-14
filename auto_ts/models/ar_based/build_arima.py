from typing import Optional
import warnings
import itertools
import operator
import copy

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from pandas.core.generic import NDFrame # type:ignore

import matplotlib.pyplot as plt  # type: ignore
import seaborn as sns  # type: ignore
sns.set(style="white", color_codes=True)

# imported ARIMA from statsmodels pkg
from statsmodels.tsa.arima_model import ARIMA # type: ignore

# helper functions
from ...utils import print_static_rmse, print_dynamic_rmse
from ...models.ar_based.param_finder import find_lowest_pq
import pdb


class BuildArima():
    def __init__(self, metric='aic', p_max=3, d_max=1, q_max=3, forecast_period=2, method='mle', verbose=0):
        """
        Automatically build an ARIMA Model
        """
        self.metric = metric
        self.p_max = p_max
        self.d_max = d_max
        self.q_max = q_max
        self.forecast_period = forecast_period
        self.method = method
        self.verbose = verbose
        self.model = None

    def fit(self, ts_df):
        """
        Build a Time Series Model using SARIMAX from statsmodels.

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

        solver = 'lbfgs'  # default

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
        ts_train = ts_df[:-self.forecast_period]
        ts_test = ts_df[-self.forecast_period:]
        if self.verbose == 1:
            print('Data Set split into train %s and test %s for Cross Validation Purposes'
                % (ts_train.shape, ts_test.shape))
        #########################################################################
        if ts_train.dtype == 'int64':
            ts_train = ts_train.astype(float)
        for d_val in range(d_min, self.d_max+1):
            print('\nDifferencing = %d' % d_val)
            results_bic = pd.DataFrame(
                index=['AR{}'.format(i) for i in range(p_min, self.p_max+1)],
                columns=['MA{}'.format(i) for i in range(q_min, self.q_max+1)]
            )
            for p_val, q_val in itertools.product(range(p_min, self.p_max+1), range(q_min, self.q_max+1)):
                if p_val == 0 and d_val == 0 and q_val == 0:
                    results_bic.loc['AR{}'.format(p_val), 'MA{}'.format(q_val)] = np.nan
                    continue
                else:
                    try:
                        model = ARIMA(ts_train, order=(p_val, d_val, q_val))
                        results = model.fit(transparams=False, method=self.method, solver=solver, disp=False)
                        results_bic.loc['AR{}'.format(p_val), 'MA{}'.format(q_val)] = eval('results.' + self.metric)
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
            if self.verbose == 1:
                _, ax = plt.subplots(figsize=(20, 10))
                ax = sns.heatmap(results_bic,
                                mask=results_bic.isnull(),
                                ax=ax,
                                annot=True,
                                fmt='.0f')
                ax.set_title(self.metric)
            results_dict[str(interim_p) + ' ' + str(interim_d) + ' ' + str(interim_q)] = interim_bic
        best_bic = min(results_dict.items(), key=operator.itemgetter(1))[1]
        best_pdq = min(results_dict.items(), key=operator.itemgetter(1))[0]
        best_p = int(best_pdq.split(' ')[0])
        best_d = int(best_pdq.split(' ')[1])
        best_q = int(best_pdq.split(' ')[2])
        print('\nBest model is: Non Seasonal ARIMA(%d,%d,%d), %s = %0.3f' % (best_p, best_d, best_q, self.metric, best_bic))
        bestmodel = ARIMA(ts_train, order=(best_p, best_d, best_q))
        print('####    Fitting best model for full data set now. Will take time... ######')
        try:
            self.model = bestmodel.fit(transparams=True, method=self.method, solver=solver, disp=False)
        except:
            self.model = bestmodel.fit(transparams=False, method=self.method, solver=solver, disp=False)
        ### this is needed for static forecasts ####################
        y_truth = ts_train[:]
        y_forecasted = self.model.predict(typ='levels')
        concatenated = pd.concat([y_truth, y_forecasted], axis=1, keys=['original', 'predicted'])
        if best_d == 0:
            #### Do this for ARIMA only ######
            ###  If there is no differencing DO NOT use predict_type since it will give an error = do not use "linear".
            print('Static Forecasts:')
            print_static_rmse(concatenated['original'].values, concatenated['predicted'].values, best_d)
            start_date = ts_df.index[-self.forecast_period]
            end_date = ts_df.index[-1]
            pred_dynamic = self.model.predict(start=start_date, end=end_date, dynamic=True)
            if self.verbose == 1:
                ax = concatenated[['original', 'predicted']][best_d:].plot()
                pred_dynamic.plot(label='Dynamic Forecast', ax=ax, figsize=(15, 5))
                print('Dynamic %d-period Forecasts:' % (self.forecast_period,))
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

            # TODO: Check if this can be changed to use predict function directly.
            start_date = ts_df.index[-self.forecast_period]
            end_date = ts_df.index[-1]
            pred_dynamic = self.model.predict(typ=pred_type, start=start_date, end=end_date, dynamic=True)
            try:
                pred_dynamic[pd.to_datetime((pred_dynamic.index-best_d).values[0])] = \
                                        y_truth[pd.to_datetime((pred_dynamic.index-best_d).values[0])]
            except:
                print('Dynamic predictions erroring but continuing...')
            pred_dynamic.sort_index(inplace=True)
            print('\nDynamic %d-period Forecasts:' % self.forecast_period)
            if self.verbose == 1:
                ax = concatenated.plot()
                pred_dynamic.plot(label='Dynamic Forecast', ax=ax, figsize=(15, 5))
                ax.set_xlabel('Date')
                ax.set_ylabel('Values')
                plt.legend()
                plt.show(block=False)
        if self.verbose == 1:
            try:
                self.model.plot_diagnostics(figsize=(16, 12))
            except:
                pass
        print(self.model.summary())

        res_frame = self.predict(simple=False)

        if self.verbose == 1:
            print('Model Forecast(s):\n', res_frame)
        rmse, norm_rmse = print_dynamic_rmse(ts_test, pred_dynamic, ts_train)
        return self.model, res_frame, rmse, norm_rmse

    def predict(
        self,
        testdata: Optional[pd.DataFrame]=None,
        forecast_period: Optional[int] = None,
        simple: bool = True) -> NDFrame:
        """
        Return the predictions
        # TODO: Check if the series can be converted to a dataframe for all models.
        :rtype cam be Pandas Series (simple), pandas dataframe (simple = False) or None
        """

        # TODO: Add processing of 'simple' argument and return type

        if testdata is not None:
            warnings.warn(
                "You have passed exogenous variables to make predictions for a ARIMA model." +
                "ARIMA models are univariate models and hence these exogenous variables will be ignored for these predictions."
            )

        # TODO: Predictions coming from ARIMA include extra information compared to SARIMAX and VAR.
        # Need to make it consistent
        # Extract the dynamic predicted and true values of our time series
        if forecast_period is None:
            # use the forecast period used during training
            forecast_period = self.forecast_period

        y_forecasted = self.model.forecast(forecast_period)


        # TODO: Check if the datetime index can be obtained as in the case of SARIMAX.
        # Currently it is just a text index, e.g. Forecast_1, ...
        if simple:
            res_frame = pd.DataFrame([
                y_forecasted[0], # Mean Forecast
                ],
                index=['mean'],
                columns=['Forecast_' + str(x) for x in range(1, forecast_period+1)]
            ).T
            res_frame = res_frame.squeeze() # Convert to a pandas series object
        else:
            res_frame = pd.DataFrame([
                y_forecasted[0], # Mean Forecast
                y_forecasted[1], # Std Error
                y_forecasted[2], # Lower and Upper CI
                ],
                index=['mean','mean_se','mean_ci'],
                columns=['Forecast_' + str(x) for x in range(1, forecast_period+1)]
            ).T

            res_frame['mean_ci_lower'] = res_frame['mean_ci'].map(lambda x: x[0])
            res_frame['mean_ci_upper'] = res_frame['mean_ci'].map(lambda x: x[1])
            res_frame.drop('mean_ci', axis=1, inplace=True)

        return res_frame
