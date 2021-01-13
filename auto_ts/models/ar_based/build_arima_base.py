from typing import Optional
import warnings
warnings.filterwarnings(action='ignore')
from abc import abstractmethod
import copy
import pdb

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from pandas.core.generic import NDFrame # type:ignore

import matplotlib.pyplot as plt # type: ignore

#from tscv import GapWalkForward # type: ignore
from sklearn.model_selection import TimeSeriesSplit

# imported SARIMAX from statsmodels pkg
from statsmodels.tsa.statespace.sarimax import SARIMAX  # type: ignore

from ..build_base import BuildBase

# helper functions
from ...utils import colorful, print_static_rmse, print_dynamic_rmse
from ...models.ar_based.param_finder import find_best_pdq_or_PDQ


class BuildArimaBase(BuildBase):
    def __init__(self, scoring, seasonality=False, seasonal_period=None, p_max=12,
            d_max=2, q_max=12, forecast_period=5, verbose=0):
        """
        Base class for building any ARIMA model
        Definitely applicable to SARIMAX and auto_arima with seasonality
        Check later if same can be reused for ARIMA (most likely yes)
        """
        super().__init__(
            scoring=scoring,
            forecast_period=forecast_period,
            verbose=verbose
        )

        self.seasonality = seasonality
        self.seasonal_period = seasonal_period
        self.p_max = p_max
        self.d_max = d_max
        self.q_max = q_max

        self.best_p = None
        self.best_d = None
        self.best_q = None
        self.best_P = None
        self.best_D = None
        self.best_Q = None


    def fit(self, ts_df: pd.DataFrame, target_col: str, cv: Optional[int]=None):
        """
        Build a Time Series Model using SARIMAX from statsmodels.
        """

        self.original_target_col = target_col
        self.original_preds = [x for x in list(ts_df) if x not in [self.original_target_col]]

        if len(self.original_preds) == 0:
            self.univariate = True
        else:
            self.univariate = False


        ##########################################
        #### Find best pdq and PDQ parameters ####
        ##########################################

        # NOTE: We use the entire dataset to compute the pdq and PDQ parameters.
        # Then we use the selected "best" parameters to check how well it
        # generalizes across the various folds (which may even be 1)

        # ## Added temporarily
        # ts_train = ts_df.iloc[:-self.forecast_period]
        # self.find_best_parameters(data = ts_train)
        auto_arima_model = self.find_best_parameters(data = ts_df)
        self.model = auto_arima_model

        if self.verbose >= 1:
            print(f"\n\nBest Parameters:")
            print(f"p: {self.best_p}, d: {self.best_d}, q: {self.best_q}")
            print(f"P: {self.best_P}, D: {self.best_D}, Q: {self.best_Q}")
            print(f"Seasonality: {self.seasonality}\nSeasonal Period: {self.seasonal_period}")

        #######################################
        #### Cross Validation across Folds ####
        #######################################

        rmse_folds = []
        norm_rmse_folds = []
        forecast_df_folds = []

        NFOLDS = self.get_num_folds_from_cv(cv)

        #########################################################################
        num_obs = ts_df.shape[0]
        if self.forecast_period <= 5:
            #### Set a minimum of 5 for the number of rows in test!
            self.forecast_period = 5
        ### In case the number of forecast_period is too high, just reduce it so it can fit into num_obs
        if NFOLDS*self.forecast_period > num_obs:
            self.forecast_period = int(num_obs/(NFOLDS+1))
            print('Lowering forecast period to %d to enable cross_validation' %self.forecast_period)
        #########################################################################
        max_trainsize = len(ts_df) - self.forecast_period
        cv = TimeSeriesSplit(n_splits=NFOLDS, max_train_size = max_trainsize)
        for fold_number, (train, test) in enumerate(cv.split(ts_df)):
            ts_train = ts_df.iloc[train]
            ts_test = ts_df.iloc[test]

            if self.verbose >= 1:
                print(f"\nFold Number: {fold_number+1} --> Train Shape: {ts_train.shape[0]} Test Shape: {ts_test.shape[0]}")

            ### this is needed for static forecasts ####################
            # TODO: Check if this needs to be fixed to pick usimg self.original_target_col
            y_truth = ts_train[:]  #  TODO: Note that this is only univariate analysis

            if len(self.original_preds) == 0:
                exog = None
            elif len(self.original_preds) == 1:
                exog = ts_test[self.original_preds[0]].values.reshape(-1, 1)
            else:
                exog = ts_test[self.original_preds].values

            y_forecasted = self.model.predict(ts_test.shape[0],exog)

            concatenated = pd.DataFrame(np.c_[ts_test[self.original_target_col].values,
                            y_forecasted], columns=['original', 'predicted'],index=ts_test.index)

            ### for SARIMAX and Auto_ARIMA, you don't have to restore differences since it predicts like actuals.###
            y_true = concatenated['original']
            y_pred = concatenated['predicted']

            if self.verbose >= 1:
                print('Static Forecasts:')
                # Since you are differencing the data, some original data points will not be available
                # Hence taking from first available value.
                print_static_rmse(y_true, y_pred, verbose=self.verbose)
                #quick_ts_plot(y_true, y_pred)

            # Extract the dynamic predicted and true values of our time series
            forecast_df = copy.deepcopy(y_forecasted)
            forecast_df_folds.append(forecast_df)


            rmse, norm_rmse = print_static_rmse(y_true, y_pred, verbose=0) ## don't print this time
            rmse_folds.append(rmse)
            norm_rmse_folds.append(norm_rmse)

            # TODO: Convert rmse_folds, rmse_norm_folds, forecasts_folds into base class attributes
            # TODO: Add gettes and seters for these class attributes.
            # This will ensure consistency across various model build types.


        # This is taking the std of entire dataset and using that to normalize
        # vs. other approach that was using std of individual folds to standardize.
        # Technically this is not correct, but in order to do Apples:Aples compatison with ML
        # (sklearn) based cross_val_score, we need to do this since we dont get individual folds
        # back for cross_val_score. If at a later point in time, we can get this, then,
        # we can revert back to dividing by individual fold std values.
        norm_rmse_folds2 = rmse_folds/ts_df[self.original_target_col].values.std()  # Same as what was there in print_dynamic_rmse()

        print(f"\nSARIMAX RMSE (all folds): {np.mean(rmse_folds):.4f}")
        print(f"SARIMAX Norm RMSE (all folds): {(np.mean(norm_rmse_folds2)*100):.0f}%\n")

        ###############################################
        #### Refit the model on the entire dataset ####
        ###############################################
        self.refit(ts_df=ts_df)

        if self.verbose >= 1:
            print(self.model.summary())

        # return self.model, forecast_df_folds, rmse_folds, norm_rmse_folds
        return self.model, forecast_df_folds, rmse_folds, norm_rmse_folds2

    def refit(self, ts_df: pd.DataFrame) -> object:
        """
        Refits an already trained model using a new dataset
        Useful when fitting to the full data after testing with cross validation
        :param ts_df The time series data to be used for fitting the model
        :type ts_df pd.DataFrame
        :rtype object
        """

        bestmodel = self.get_best_model(ts_df)

        print(colorful.BOLD + 'Refitting data with previously found best parameters' + colorful.END)
        try:
            self.model = bestmodel.fit(disp=False)
            print('    Best %s metric = %0.1f' % (self.scoring, eval('self.model.' + self.scoring)))
        except Exception as e:
            print(e)

        return self

    @abstractmethod
    def find_best_parameters(self, data: pd.DataFrame):
        """
        Given a dataset, finds the best parameters using the settings in the class
        Need to set the following parameters in the child class
        self.best_p, self.best_d, self.best_q
        self.best_P, self.best_D, self.best_Q
        """



    def get_best_model(self, data: pd.DataFrame):
        """
        Returns the 'unfit' SARIMAX model with the given dataset and the
        selected best parameters. This can be used to fit or refit the model.
        """

        # In order to get forecasts to be in the same value ranges of the orig_endogs, you
        # must  set the simple_differencing = False and the start_params to be the same as ARIMA.
        # That is the only way to ensure that the output of this model iscomparable to other ARIMA models

        if not self.seasonality:
            if self.univariate:
                bestmodel = SARIMAX(
                    endog=data[self.original_target_col],
                    # exog=data[self.original_preds], ###if it is univariate, no preds needed
                    order=(self.best_p, self.best_d, self.best_q),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                    trend='ct',
                    start_params=[0, 0, 0, 1],
                    simple_differencing=False)
            else:
                bestmodel = SARIMAX(
                    endog=data[self.original_target_col],
                    exog=data[self.original_preds], ## if it is multivariate, preds are needed
                    order=(self.best_p, self.best_d, self.best_q),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                    trend='ct',
                    start_params=[0, 0, 0, 1],
                    simple_differencing=False)
        else:
            if self.univariate:
                bestmodel = SARIMAX(
                    endog=data[self.original_target_col],
                    # exog=data[self.original_preds], ### if univariate, no preds are needed
                    order=(self.best_p, self.best_d, self.best_q),
                    seasonal_order=(self.best_P, self.best_D, self.best_Q, self.seasonal_period),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                    trend='ct',
                    start_params=[0, 0, 0, 1],
                    simple_differencing=False
                )
            else:
                bestmodel = SARIMAX(
                    endog=data[self.original_target_col],
                    exog=data[self.original_preds], ### if multivariate, preds are needed
                    order=(self.best_p, self.best_d, self.best_q),
                    seasonal_order=(self.best_P, self.best_D, self.best_Q, self.seasonal_period),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                    trend='ct',
                    start_params=[0, 0, 0, 1],
                    simple_differencing=False
                )

        return bestmodel

    def predict(
        self,
        testdata: Optional[pd.DataFrame]=None,
        forecast_period: Optional[int] = None,
        simple: bool = True) -> NDFrame:
        """
        Return the predictions
        """
        # Extract the dynamic predicted and true values of our time series
        if self.univariate:
            if isinstance(testdata, pd.DataFrame) or isinstance(testdata, pd.Series):
                # use the forecast period used during training
                forecast_period = testdata.shape[0]
                self.forecast_period = testdata.shape[0]
        else:
            if testdata is None:
                raise ValueError("SARIMAX needs testdata to make predictions, but this was not provided. Please provide to proceed.")
                forecast_period = self.forecast_period
            elif isinstance(testdata, pd.DataFrame) or isinstance(testdata, pd.Series):
                if forecast_period != testdata.shape[0]:
                    warnings.warn("Forecast Period is not equal to the number of observations in testdata. The forecast period will be assumed to be the number of observations in testdata.")
                forecast_period = testdata.shape[0]
                self.forecast_period = forecast_period
                try:
                    testdata = testdata[self.original_preds]
                except Exception as e:
                    print(e)
                    print("Model was trained with train dataframe. Please make sure you are passing a test data frame.")
                    return
            elif isinstance(testdata, int):
                if forecast_period != testdata:
                    print("Forecast Period is not equal to the number of observations in testdata. The forecast period will be assumed to be the number of observations in testdata.")

                forecast_period = testdata
                self.forecast_period = forecast_period

        if self.univariate:
            res = self.model.get_forecast(self.forecast_period)
        else:
            if isinstance(testdata, pd.DataFrame) or isinstance(testdata, pd.Series):
                res = self.model.get_forecast(self.forecast_period, exog=testdata)
            else:
                try:
                    res = self.model.get_forecast(self.forecast_period)
                except Exception as e:
                    print(e)
                    print("Model was trained with train dataframe. Please make sure you are passing a test data frame.")
                    return

        res_frame = res.summary_frame()
        res_frame.rename(columns = {'mean':'yhat'}, inplace=True)

        if simple:
            res_frame = res_frame['yhat']
            res_frame = res_frame.squeeze() # Convert to a pandas series object
        else:
            # Pass as is
            pass

        return res_frame
