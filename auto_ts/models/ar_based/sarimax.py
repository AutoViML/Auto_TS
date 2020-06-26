from typing import Optional
import warnings
import copy

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from pandas.core.generic import NDFrame # type:ignore

import matplotlib.pyplot as plt # type: ignore

# imported SARIMAX from statsmodels pkg
from statsmodels.tsa.statespace.sarimax import SARIMAX  # type: ignore

# helper functions
from ...utils import colorful, print_static_rmse, print_dynamic_rmse
from ...models.ar_based.param_finder import find_best_pdq_or_PDQ


class BuildSarimax():
    def __init__(self, metric, seasonality=False, seasonal_period=None, p_max=12, d_max=2, q_max=12, forecast_period=2, verbose=0):
        """
        Automatically build a SARIMAX Model
        """
        self.metric = metric
        self.seasonality = seasonality
        self.seasonal_period = seasonal_period
        self.p_max = p_max
        self.d_max = d_max
        self.q_max = q_max
        self.forecast_period = forecast_period
        self.verbose = verbose
        self.model = None
       

    def fit(self, ts_df: pd.DataFrame, target_col: str):
        """
        Build a Time Series Model using SARIMAX from statsmodels.
        """

        self.original_target_col = target_col
        self.original_preds = [x for x in list(ts_df) if x not in [self.original_target_col]]

        if len(self.original_preds) == 0:
            self.univariate = True
        else:
            self.univariate = False

        ############ Split the data set into train and test for Cross Validation Purposes ########
        ts_train = ts_df.iloc[:-self.forecast_period]
        ts_test = ts_df.iloc[-self.forecast_period:]

        if self.verbose == 1:
            print('Data Set split into train %s and test %s for Cross Validation Purposes'
                                % (ts_train.shape, ts_test.shape))
        ############# Now find the best pdq and PDQ parameters for the model #################
        if not self.seasonality:
            print('Building a Non Seasonal Model...')
            print('\nFinding best Non Seasonal Parameters:')
            # TODO: Check if we need to also pass the exogenous variables here and 
            # change the functionality of find_best_pdq_or_PDQ to incorporate these 
            # exogenoug variables.
            best_p, best_d, best_q, best_bic, seasonality = find_best_pdq_or_PDQ(
                # ts_train,
                ts_train[self.original_target_col],
                self.metric,
                self.p_max, self.d_max, self.q_max, non_seasonal_pdq=None,
                seasonal_period=None, seasonality=False, verbose=self.verbose)
            print('\nBest model is: Non Seasonal SARIMAX(%d,%d,%d), %s = %0.3f' % (best_p, best_d,
                                                            best_q,self.metric, best_bic))
            #### In order to get forecasts to be in the same value ranges of the orig_endogs,
            #### you must  set the simple_differencing = False and the start_params to be the
            #### same as ARIMA.
            #### THat is the only way to ensure that the output of this model is
            #### comparable to other ARIMA models
            if self.univariate:
                bestmodel = SARIMAX(
                    endog=ts_train[self.original_target_col],
                    # exog=ts_train[self.original_preds],
                    order=(best_p, best_d, best_q),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                    trend='ct',
                    start_params=[0, 0, 0, 1],
                    simple_differencing=False)
            else:
                bestmodel = SARIMAX(
                    endog=ts_train[self.original_target_col],
                    exog=ts_train[self.original_preds],
                    order=(best_p, best_d, best_q),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                    trend='ct',
                    start_params=[0, 0, 0, 1],
                    simple_differencing=False)
        else:
            print(colorful.BOLD + 'Building a Seasonal Model...'+colorful.END)
            print(colorful.BOLD + '\n    Finding best Non-Seasonal pdq Parameters:' + colorful.END)
            # TODO: Check if we need to also pass the exogenous variables here and 
            # change the functionality of find_best_pdq_or_PDQ to incorporate these 
            # exogenoug variables.
            best_p, best_d, best_q, best_bic, seasonality = find_best_pdq_or_PDQ(
                # ts_train,
                ts_train[self.original_target_col],
                self.metric,
                self.p_max, self.d_max, self.q_max,
                non_seasonal_pdq=None,
                seasonal_period=None,
                seasonality=False,verbose=self.verbose)
            print(colorful.BOLD + '\n    Finding best Seasonal PDQ Model Parameters:' + colorful.END)
            # TODO: Check if we need to also pass the exogenous variables here and 
            # change the functionality of find_best_pdq_or_PDQ to incorporate these 
            # exogenoug variables.
            best_P, best_D, best_Q, best_bic, seasonality = find_best_pdq_or_PDQ(
                # ts_train,
                ts_train[self.original_target_col],
                self.metric,
                self.p_max, self.d_max, self.q_max,
                non_seasonal_pdq=(best_p, best_d, best_q),
                seasonal_period=self.seasonal_period,
                seasonality=True, verbose=self.verbose)
            
                        
            if seasonality:
                print('\nBest model is a Seasonal SARIMAX(%d,%d,%d)*(%d,%d,%d,%d), %s = %0.3f' % (
                                                best_p, best_d, best_q, best_P,
                                                best_D, best_Q, self.seasonal_period, self.metric, best_bic))
                #### In order to get forecasts to be in the same value ranges of the orig_endogs,
                #### you must set the simple_differencing =False and the start_params to be
                #### the same as ARIMA.
                #### THat is the only way to ensure that the output of this model is
                #### comparable to other ARIMA models
                if self.univariate:
                    bestmodel = SARIMAX(
                        endog=ts_train[self.original_target_col],
                        # exog=ts_train[self.original_preds],
                        order=(best_p, best_d, best_q),
                        seasonal_order=(best_P, best_D, best_Q, self.seasonal_period),
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                        simple_differencing=False, trend='ct',
                        start_params=[0, 0, 0, 1])
                else:
                    bestmodel = SARIMAX(
                        endog=ts_train[self.original_target_col],
                        exog=ts_train[self.original_preds],
                        order=(best_p, best_d, best_q),
                        seasonal_order=(best_P, best_D, best_Q, self.seasonal_period),
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                        simple_differencing=False, trend='ct',
                        start_params=[0, 0, 0, 1])
            else:
                print('\nBest model is a Non Seasonal SARIMAX(%d,%d,%d)' % (
                                                    best_p, best_d, best_q))
                #### In order to get forecasts to be in the same value ranges of the orig_endogs,
                #### you must set the simple_differencing =False and the start_params to be
                #### the same as ARIMA.
                #### THat is the only way to ensure that the output of this model is
                #### comparable to other ARIMA models
                if self.univariate:
                    bestmodel = SARIMAX(
                        endog=ts_train[self.original_target_col],
                        # exog=ts_train[self.original_preds],
                        order=(best_p, best_d, best_q),
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                        trend='ct',
                        start_params=[0, 0, 0, 1],
                        simple_differencing=False)
                else:
                    bestmodel = SARIMAX(
                        endog=ts_train[self.original_target_col],
                        exog=ts_train[self.original_preds],
                        order=(best_p, best_d, best_q),
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                        trend='ct',
                        start_params=[0, 0, 0, 1],
                        simple_differencing=False)
        
        print(colorful.BOLD + 'Fitting best SARIMAX model for full data set'+colorful.END)
        try:
            self.model = bestmodel.fit()
            print('    Best %s metric = %0.1f' % (self.metric, eval('self.model.' + self.metric)))
        except:
            print('Error: Getting Singular Matrix. Please try using other PDQ parameters or turn off Seasonality')
            return bestmodel, None, np.inf, np.inf
        
        if self.verbose == 1:
            try:
                self.model.plot_diagnostics(figsize=(16, 12))
            except:
                print('Error: SARIMAX plot diagnostic. Continuing...')
        
        ### this is needed for static forecasts ####################
        y_truth = ts_train[:]  # Note that this is only univariate analysis

        if self.univariate:
            y_forecasted = self.model.predict(dynamic=False)
        else:
            y_forecasted = self.model.predict(dynamic=False, exog=ts_test[self.original_preds])

        concatenated = pd.concat([y_truth, y_forecasted], axis=1, keys=['original', 'predicted'])

        ### for SARIMAX, you don't have to restore differences since it predicts like actuals.###
        if self.verbose == 1:
            print('Static Forecasts:')
            # Since you are differencing the data, some original data points will not be available
            # Hence taking from first available value.
            print_static_rmse(concatenated['original'].values[best_d:],
                              concatenated['predicted'].values[best_d:],
                              verbose=self.verbose)
        
        ########### Dynamic One Step Ahead Forecast ###########################
        ### Dynamic Forecats are a better representation of true predictive power
        ## since they only use information from the time series up to a certain point,
        ## and after that, forecasts are generated using values from previous forecasted
        ## time points.
        #################################################################################
        # Now do dynamic forecast plotting for the last X steps of the data set ######

        if self.verbose == 1:
            ax = concatenated[['original', 'predicted']][best_d:].plot(figsize=(16, 12))
            startdate = ts_df.index[-self.forecast_period-1]
            pred_dynamic = self.model.get_prediction(start=startdate, dynamic=True, full_results=True)
            pred_dynamic_ci = pred_dynamic.conf_int()
            pred_dynamic.predicted_mean.plot(label='Dynamic Forecast', ax=ax)
            try:
                ax.fill_between(pred_dynamic_ci.index, pred_dynamic_ci.iloc[:, 0],
                                pred_dynamic_ci.iloc[:, 1], color='k', alpha=.25)
                ax.fill_betweenx(ax.get_ylim(), startdate, ts_train.index[-1], alpha=.1, zorder=-1)
            except:
                pass
            ax.set_xlabel('Date')
            ax.set_ylabel('Levels')
            plt.legend()
            plt.show(block=False)
        
        # Extract the dynamic predicted and true values of our time series
        res_df = self.predict(X_exogen=ts_test[self.original_preds], simple=False)
        
        if self.verbose == 1:
            print(self.model.summary())
        print('Dynamic %d-Period Forecast:' % (self.forecast_period))
        rmse, norm_rmse = print_dynamic_rmse(ts_test[self.original_target_col], res_df['mean'].values, ts_train[self.original_target_col])
        return self.model, res_df, rmse, norm_rmse

    def predict(
        self,
        X_exogen: Optional[pd.DataFrame]=None,
        forecast_period: Optional[int] = None,
        simple: bool = True) -> NDFrame:
        """
        Return the predictions
        """
        # Extract the dynamic predicted and true values of our time series
        
        if self.univariate:
            if forecast_period is None:
                # use the forecast period used during training
                forecast_period = self.forecast_period
        else:
            if X_exogen is None:
                raise ValueError("SARIMAX needs X_exogen to make predictions, but this was not provided. Please provide to proceed.")
                        
            if forecast_period != X_exogen.shape[0]:
                warnings.warn("Forecast Period is not equal to the number of observations in X_exogen. The forecast period will be assumed to be the number of observations in X_exogen.")
            
            forecast_period = X_exogen.shape[0]
        
        if self.univariate:
            res = self.model.get_forecast(forecast_period)
        else:
            res = self.model.get_forecast(forecast_period, exog=X_exogen)

        res_frame = res.summary_frame()

        if simple:
            res_frame = res_frame['mean']
            res_frame = res_frame.squeeze() # Convert to a pandas series object
        else:
            # Pass as is
            pass
            
        return res_frame
        


