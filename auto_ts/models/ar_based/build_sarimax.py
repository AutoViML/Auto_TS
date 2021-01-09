import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from pandas.core.generic import NDFrame # type:ignore

import matplotlib.pyplot as plt # type: ignore

#from tscv import GapWalkForward # type: ignore

# imported SARIMAX from statsmodels pkg
from statsmodels.tsa.statespace.sarimax import SARIMAX  # type: ignore

from ..build_base import BuildBase
from .build_arima_base import BuildArimaBase

# helper functions
from ...utils import colorful, print_static_rmse, print_dynamic_rmse
from ...models.ar_based.param_finder import find_best_pdq_or_PDQ


# class BuildSarimax(BuildBase):
class BuildSarimax(BuildArimaBase):
    # def __init__(self, scoring, seasonality=False, seasonal_period=None, p_max=12, d_max=2, q_max=12, forecast_period=2, verbose=0):
    #     """
    #     Automatically build a SARIMAX Model
    #     """
    #     super().__init__(
    #         scoring=scoring,
    #         forecast_period=forecast_period,
    #         verbose=verbose
    #     )

    #     self.seasonality = seasonality
    #     self.seasonal_period = seasonal_period
    #     self.p_max = p_max
    #     self.d_max = d_max
    #     self.q_max = q_max

    #     self.best_p = None
    #     self.best_d = None
    #     self.best_q = None
    #     self.best_P = None
    #     self.best_D = None
    #     self.best_Q = None


    # def fit(self, ts_df: pd.DataFrame, target_col: str, cv: Optional[int]=None):
    #     """
    #     Build a Time Series Model using SARIMAX from statsmodels.
    #     """

    #     self.original_target_col = target_col
    #     self.original_preds = [x for x in list(ts_df) if x not in [self.original_target_col]]

    #     if len(self.original_preds) == 0:
    #         self.univariate = True
    #     else:
    #         self.univariate = False


    #     ##########################################
    #     #### Find best pdq and PDQ parameters ####
    #     ##########################################

    #     # NOTE: We use the entire dataset to compute the pdq and PDQ parameters.
    #     # Then we use the selected "best" parameters to check how well it
    #     # generalizes across the various folds (which may even be 1)

    #     # ## Added temporarily
    #     # ts_train = ts_df.iloc[:-self.forecast_period]
    #     # self.find_best_parameters(data = ts_train)
    #     self.find_best_parameters(data = ts_df)

    #     if self.verbose >= 1:
    #         print(f"\n\nBest Parameters:")
    #         print(f"p: {self.best_p}, d: {self.best_d}, q: {self.best_q}")
    #         print(f"P: {self.best_P}, D: {self.best_D}, Q: {self.best_Q}")
    #         print(f"Seasonality: {self.seasonality} Seasonal Period: {self.seasonal_period}")


    #     #######################################
    #     #### Cross Validation across Folds ####
    #     #######################################

    #     rmse_folds = []
    #     norm_rmse_folds = []
    #     forecast_df_folds = []

    #     NFOLDS = self.get_num_folds_from_cv(cv)
    #     cv = GapWalkForward(n_splits=NFOLDS, gap_size=0, test_size=self.forecast_period)
    #     for fold_number, (train, test) in enumerate(cv.split(ts_df)):
    #         ts_train = ts_df.iloc[train]
    #         ts_test = ts_df.iloc[test]

    #         if self.verbose >= 1:
    #             print(f"\n\nFold Number: {fold_number+1} --> Train Shape: {ts_train.shape} Test Shape: {ts_test.shape}")


    #         #########################################
    #         #### Define the model with fold data ####
    #         #########################################

    #         bestmodel = self.get_best_model(ts_train)

    #         ######################################
    #         #### Fit the model with fold data ####
    #         ######################################

    #         if self.verbose >= 1:
    #             print(colorful.BOLD + 'Fitting best SARIMAX model' + colorful.END)

    #         try:
    #             self.model = bestmodel.fit(disp=False)
    #             if self.verbose >= 1:
    #                 print('    Best %s metric = %0.1f' % (self.scoring, eval('self.model.' + self.scoring)))
    #         except Exception as e:
    #             print(e)
    #             print('Error: Getting Singular Matrix. Please try using other PDQ parameters or turn off Seasonality')
    #             return bestmodel, None, np.inf, np.inf

    #         if self.verbose >= 1:
    #             try:
    #                 self.model.plot_diagnostics(figsize=(16, 12))
    #             except:
    #                 print('Error: SARIMAX plot diagnostic. Continuing...')

    #         ### this is needed for static forecasts ####################
    #         # TODO: Check if this needs to be fixed to pick usimg self.original_target_col
    #         y_truth = ts_train[:]  #  TODO: Note that this is only univariate analysis

    #         if self.univariate:
    #             y_forecasted = self.model.predict(dynamic=False)
    #         else:
    #             y_forecasted = self.model.predict(dynamic=False, exog=ts_test[self.original_preds])

    #         concatenated = pd.concat([y_truth, y_forecasted], axis=1, keys=['original', 'predicted'])

    #         ### for SARIMAX, you don't have to restore differences since it predicts like actuals.###
    #         if self.verbose >= 1:
    #             print('Static Forecasts:')
    #             # Since you are differencing the data, some original data points will not be available
    #             # Hence taking from first available value.
    #             print_static_rmse(
    #                 concatenated['original'].values[self.best_d:],
    #                 concatenated['predicted'].values[self.best_d:],
    #                 verbose=self.verbose
    #             )

    #         ########### Dynamic One Step Ahead Forecast ###########################
    #         ### Dynamic Forecats are a better representation of true predictive power
    #         ## since they only use information from the time series up to a certain point,
    #         ## and after that, forecasts are generated using values from previous forecasted
    #         ## time points.
    #         #################################################################################
    #         # Now do dynamic forecast plotting for the last X steps of the data set ######

    #         if self.verbose >= 1:
    #             ax = concatenated[['original', 'predicted']][self.best_d:].plot(figsize=(16, 12))
    #             startdate = ts_df.index[-self.forecast_period-1]
    #             pred_dynamic = self.model.get_prediction(start=startdate, dynamic=True, full_results=True)
    #             pred_dynamic_ci = pred_dynamic.conf_int()
    #             pred_dynamic.predicted_mean.plot(label='Dynamic Forecast', ax=ax)
    #             try:
    #                 ax.fill_between(pred_dynamic_ci.index, pred_dynamic_ci.iloc[:, 0],
    #                                 pred_dynamic_ci.iloc[:, 1], color='k', alpha=.25)
    #                 ax.fill_betweenx(ax.get_ylim(), startdate, ts_train.index[-1], alpha=.1, zorder=-1)
    #             except:
    #                 pass
    #             ax.set_xlabel('Date')
    #             ax.set_ylabel('Levels')
    #             plt.legend()
    #             plt.show(block=False)

    #         # Extract the dynamic predicted and true values of our time series
    #         forecast_df = self.predict(testdata=ts_test[self.original_preds], simple=False)
    #         forecast_df_folds.append(forecast_df)

    #         # Extract Metrics
    #         if self.verbose >= 1:
    #             print('Dynamic %d-Period Forecast:' % (self.forecast_period))

    #         rmse, norm_rmse = print_dynamic_rmse(ts_test[self.original_target_col], forecast_df['mean'].values, ts_train[self.original_target_col], toprint=self.verbose)
    #         rmse_folds.append(rmse)
    #         norm_rmse_folds.append(norm_rmse)

    #         # TODO: Convert rmse_folds, rmse_norm_folds, forecasts_folds into base class attributes
    #         # TODO: Add gettes and seters for these class attributes.
    #         # This will ensure consistency across various model build types.


    #     # This is taking the std of entire dataset and using that to normalize
    #     # vs. other approach that was using std of individual folds to stansardize.
    #     # Technically this is not correct, but in order to do Apples:Aples compatison with ML
    #     # (sklearn) based cross_val_score, we need to do this since we dont get indicidual folds
    #     # back for cross_val_score. If at a later point in time, we can get this, then,
    #     # we can revert back to dividing by individual fold std values.
    #     norm_rmse_folds2 = rmse_folds/ts_df[self.original_target_col].values.std()  # Same as what was there in print_dynamic_rmse()

    #     # print(f"SARIMAX Norm RMSE (Original): {norm_rmse_folds}")
    #     # print(f"SARIMAX Norm RMSE (New): {norm_rmse_folds2}")

    #     ###############################################
    #     #### Refit the model on the entire dataset ####
    #     ###############################################
    #     self.refit(ts_df=ts_df)

    #     if self.verbose >= 1:
    #         print(self.model.summary())

    #     # return self.model, forecast_df_folds, rmse_folds, norm_rmse_folds
    #     return self.model, forecast_df_folds, rmse_folds, norm_rmse_folds2

    # def refit(self, ts_df: pd.DataFrame) -> object:
    #     """
    #     Refits an already trained model using a new dataset
    #     Useful when fitting to the full data after testing with cross validation
    #     :param ts_df The time series data to be used for fitting the model
    #     :type ts_df pd.DataFrame
    #     :rtype object
    #     """

    #     bestmodel = self.get_best_model(ts_df)

    #     print(colorful.BOLD + 'Refitting data with previously found best parameters' + colorful.END)
    #     try:
    #         self.model = bestmodel.fit(disp=False)
    #         print('    Best %s metric = %0.1f' % (self.scoring, eval('self.model.' + self.scoring)))
    #     except Exception as e:
    #         print(e)

    #     return self


    def find_best_parameters(self, data: pd.DataFrame):
        """
        Given a dataset, finds the best parameters using the settings in the class
        """

        if not self.seasonality:
            if self.verbose >= 1:
                print('Building a Non Seasonal Model...')
                print('\nFinding best Non Seasonal Parameters:')
            # TODO: Check if we need to also pass the exogenous variables here and
            # change the functionality of find_best_pdq_or_PDQ to incorporate these
            # exogenoug variables.
            self.best_p, self.best_d, self.best_q, best_bic, _ = find_best_pdq_or_PDQ(
                ts_df=data[self.original_target_col],
                scoring=self.scoring,
                p_max=self.p_max, d_max=self.d_max, q_max=self.q_max,
                non_seasonal_pdq=None,
                seasonal_period=None,
                seasonality=False,
                verbose=self.verbose
            )

            if self.verbose >= 1:
                print('\nBest model is: Non Seasonal SARIMAX(%d,%d,%d), %s = %0.3f' % (
                    self.best_p, self.best_d, self.best_q, self.scoring, best_bic))
        else:
            if self.verbose >= 1:
                print(colorful.BOLD + 'Building a Seasonal Model...'+colorful.END)
                print(colorful.BOLD + '\n    Finding best Non-Seasonal pdq Parameters:' + colorful.END)
            # TODO: Check if we need to also pass the exogenous variables here and
            # change the functionality of find_best_pdq_or_PDQ to incorporate these
            # exogenoug variables.
            self.best_p, self.best_d, self.best_q, _, _ = find_best_pdq_or_PDQ(
                ts_df=data[self.original_target_col],
                scoring=self.scoring,
                p_max=self.p_max, d_max=self.d_max, q_max=self.q_max,
                non_seasonal_pdq=None,  # we need to figure this out ...
                seasonal_period=None,
                seasonality=False,  # setting seasonality = False for p, d, q
                verbose=self.verbose
            )

            if self.verbose >= 1:
                print(colorful.BOLD + '\n    Finding best Seasonal PDQ Model Parameters:' + colorful.END)
            # TODO: Check if we need to also pass the exogenous variables here and
            # change the functionality of find_best_pdq_or_PDQ to incorporate these
            # exogenoug variables.
            self.best_P, self.best_D, self.best_Q, best_bic, self.seasonality = find_best_pdq_or_PDQ(
                ts_df=data[self.original_target_col],
                scoring=self.scoring,
                p_max=self.p_max, d_max=self.d_max, q_max=self.q_max,
                non_seasonal_pdq=(self.best_p, self.best_d, self.best_q), # found previously ...
                seasonal_period=self.seasonal_period,  # passing seasonal period
                seasonality=True,  # setting seasonality = True for P, D, Q
                verbose=self.verbose
            )

            if self.seasonality:
                if self.verbose >= 1:
                    print('\nBest model is a Seasonal SARIMAX(%d,%d,%d)*(%d,%d,%d,%d), %s = %0.3f' % (
                        self.best_p, self.best_d, self.best_q,
                        self.best_P, self.best_D, self.best_Q,
                        self.seasonal_period, self.scoring, best_bic))
            else:
                if self.verbose >= 1:
                    print('\nEven though seasonality has been set to True, the best model is a Non Seasonal SARIMAX(%d,%d,%d)' % (
                        self.best_p, self.best_d, self.best_q))





    # def get_best_model(self, data: pd.DataFrame):
    #     """
    #     Returns the 'unfit' SARIMAX model with the given dataset and the
    #     selected best parameters. This can be used to fit or refit the model.
    #     """

    #     # In order to get forecasts to be in the same value ranges of the orig_endogs, you
    #     # must  set the simple_differencing = False and the start_params to be the same as ARIMA.
    #     # That is the only way to ensure that the output of this model iscomparable to other ARIMA models

    #     if not self.seasonality:
    #         if self.univariate:
    #             bestmodel = SARIMAX(
    #                 endog=data[self.original_target_col],
    #                 # exog=data[self.original_preds],
    #                 order=(self.best_p, self.best_d, self.best_q),
    #                 enforce_stationarity=False,
    #                 enforce_invertibility=False,
    #                 trend='ct',
    #                 start_params=[0, 0, 0, 1],
    #                 simple_differencing=False)
    #         else:
    #             bestmodel = SARIMAX(
    #                 endog=data[self.original_target_col],
    #                 exog=data[self.original_preds],
    #                 order=(self.best_p, self.best_d, self.best_q),
    #                 enforce_stationarity=False,
    #                 enforce_invertibility=False,
    #                 trend='ct',
    #                 start_params=[0, 0, 0, 1],
    #                 simple_differencing=False)
    #     else:
    #         if self.univariate:
    #             bestmodel = SARIMAX(
    #                 endog=data[self.original_target_col],
    #                 # exog=data[self.original_preds],
    #                 order=(self.best_p, self.best_d, self.best_q),
    #                 seasonal_order=(self.best_P, self.best_D, self.best_Q, self.seasonal_period),
    #                 enforce_stationarity=False,
    #                 enforce_invertibility=False,
    #                 trend='ct',
    #                 start_params=[0, 0, 0, 1],
    #                 simple_differencing=False
    #             )
    #         else:
    #             bestmodel = SARIMAX(
    #                 endog=data[self.original_target_col],
    #                 exog=data[self.original_preds],
    #                 order=(self.best_p, self.best_d, self.best_q),
    #                 seasonal_order=(self.best_P, self.best_D, self.best_Q, self.seasonal_period),
    #                 enforce_stationarity=False,
    #                 enforce_invertibility=False,
    #                 trend='ct',
    #                 start_params=[0, 0, 0, 1],
    #                 simple_differencing=False
    #             )

    #     return bestmodel

    # def predict(
    #     self,
    #     testdata: Optional[pd.DataFrame]=None,
    #     forecast_period: Optional[int] = None,
    #     simple: bool = True) -> NDFrame:
    #     """
    #     Return the predictions
    #     """
    #     # Extract the dynamic predicted and true values of our time series

    #     if self.univariate:
    #         if forecast_period is None:
    #             # use the forecast period used during training
    #             forecast_period = self.forecast_period
    #     else:
    #         if testdata is None:
    #             raise ValueError("SARIMAX needs testdata to make predictions, but this was not provided. Please provide to proceed.")

    #         if forecast_period != testdata.shape[0]:
    #             warnings.warn("Forecast Period is not equal to the number of observations in testdata. The forecast period will be assumed to be the number of observations in testdata.")

    #         forecast_period = testdata.shape[0]

    #         try:
    #             testdata = testdata[self.original_preds]
    #         except Exception as e:
    #             print(e)
    #             raise ValueError("Some exogenous columns that were used during training are missing in testdata. Please make sure you are passing the correct exogenous columns.")

    #     if self.univariate:
    #         res = self.model.get_forecast(forecast_period)
    #     else:
    #         res = self.model.get_forecast(forecast_period, exog=testdata)

    #     res_frame = res.summary_frame()

    #     if simple:
    #         res_frame = res_frame['mean']
    #         res_frame = res_frame.squeeze() # Convert to a pandas series object
    #     else:
    #         # Pass as is
    #         pass

    #     return res_frame
