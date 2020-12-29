"""Module to Build a Prphet Model
"""
from typing import Optional
import logging
import copy
import time
import numpy as np

import pandas as pd # type: ignore
from pandas.core.generic import NDFrame # type:ignore
import pdb

import matplotlib.pyplot as plt # type: ignore

from fbprophet import Prophet # type: ignore
from fbprophet.diagnostics import cross_validation
from fbprophet.diagnostics import performance_metrics
from fbprophet.plot import plot_cross_validation_metric

from .build_base import BuildBase

# helper functions
from ..utils import print_dynamic_rmse
from ..utils.logging import SuppressStdoutStderr

#### Suppress INFO messages from FB Prophet!

logging.getLogger('fbprophet').setLevel(logging.WARNING)


class BuildProphet(BuildBase):
    """Class to build a Prophet Model
    """
    def __init__(
            self, forecast_period, time_interval, scoring, verbose, conf_int, holidays, growth, seasonality
        ):
        """
        Automatically build a Prophet Model
        """
        super().__init__(
            scoring=scoring,
            forecast_period=forecast_period,
            verbose=verbose
        )

        self.time_interval = time_interval
        self.conf_int = conf_int
        self.holidays = holidays
        self.growth = growth
        self.seasonality = seasonality
        if self.time_interval == 'weeks':
            weekly_seasonality =  seasonality
        elif self.time_interval == 'years':
            yearly_seasonality = seasonality
        elif self.time_interval == 'days':
            daily_seasonality = seasonality
        else:
            yearly_seasonality=False
            weekly_seasonality=False
            daily_seasonality=False
        self.model = Prophet(
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality,
            interval_width=self.conf_int,
            holidays = self.holidays,
            growth = self.growth)
        self.univariate = None

    def fit(self, ts_df: pd.DataFrame, target_col: str, cv: Optional[int], time_col: str) -> object:
        """
        Fits the model to the data

        :param ts_df The time series data to be used for fitting the model
        :type ts_df pd.DataFrame

        :param target_col The column name of the target time series that needs to be modeled.
        All other columns will be considered as exogenous variables (if applicable to method)
        :type target_col str

        :param cv: Number of folds to use for cross validation.
        Number of observations in the Validation set for each fold = forecast period
        If None, a single fold is used
        :type cv Optional[int]

        :param time_col: Name of the time column in the dataset (needed by Prophet)
        Time column can also be the index, in which case, this would be the name of the index
        :type time_col str

        :rtype object
        """
        # use all available threads/cores

        self.time_col = time_col
        self.original_target_col = target_col
        self.original_preds = [x for x in list(ts_df) if x not in [self.original_target_col]]

        if len(self.original_preds) == 0:
            self.univariate = True
        else:
            self.univariate = False

        # print(f"Prophet Is Univariate: {self.univariate}")

        ts_df = copy.deepcopy(ts_df)

        ##### if you are going to use matplotlib with prophet data, it gives an error unless you do this.
        pd.plotting.register_matplotlib_converters()

        #### You have to import Prophet if you are going to build a Prophet model #############
        actual = 'y'
        timecol = 'ds'

        data = self.prep_col_names_for_prophet(ts_df=ts_df, test=False)

        if self.univariate:
            dft = data[[timecol, actual]]
        else:
            dft = data[[timecol, actual] + self.original_preds]

        ##### For most Financial time series data, 80% conf interval is enough...
        if self.verbose >= 1:
            print('    Fit-Predict data (shape=%s) with Confidence Interval = %0.2f...' % (dft.shape, self.conf_int))
        ### Make Sure you lower your desired interval width from the normal 95% to a more realistic 80%
        start_time = time.time()

        if self.univariate is False:
            for name in self.original_preds:
                self.model.add_regressor(name)

        print("  Starting Prophet Fit")
        with SuppressStdoutStderr():
            self.model.fit(dft)
        print("  End of Prophet Fit")

        num_obs = dft.shape[0]
        NFOLDS = self.get_num_folds_from_cv(cv)

        if self.verbose >= 2:
            print(f"NumObs: {num_obs}")
            print(f"NFOLDS: {NFOLDS}")

        if self.time_interval in ['days', 'weeks', 'months', 'years']:
            total_days = (dft['ds'].max() - dft['ds'].min()).days
        else:
            ### if time period is shorter than days, it must be calculated in hours or mins.
            total_days = (dft['ds'].max() - dft['ds'].min()).days

        if self.verbose >= 2:
            print("Variables used for calculating initial, horizon, period...")
            print(f"Forcast Period: {self.forecast_period}")
            print(f"Max Date: {dft['ds'].max()}")
            print(f"Horizon Start: {dft.iloc[-self.forecast_period]['ds']}")


        #########################################################################################
        # NOTE: This change to the FB recommendation will cause the cv folds from facebook to
        # be incompatible with the folds from the other models (in terms of periods of evaluation
        # as well as number of observations in each period). Hence the final comparison will
        # be biased since it will not compare the same folds.

        # The original implementation was giving issues under certain conditions, hence this change
        # to FB recommendation has been made as a temporary (short term) fix.
        # The root cause issue will need to be fixed eventually at a later point.
        #########################################################################################

        ### Prophet's Time Interval translates into frequency based on the following pandas date_range alias:
        #  Link: https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases
        ## This is done using the get_prophet_time_interval() function later.

        ## we will be using the recommended defaults for these form FB Prophet page
        #horizon_days = (dft['ds'].max() - dft.iloc[-forecast_start]['ds']).days
        #horizon_days = min(365, (dft['ds'].max() - dft.iloc[-(self.forecast_period+1)]['ds']).days )
        ### if the forecast_period is given as 5 and the time_interval is Monthly, then horizon = 5 months
        time_interval_days = {
                'months': 30,
                'weeks': 7,
                'days': 1,
                'semi': 182,
                'annual':365,
                'qtr': 91,
                }
        ## set the time period based on days since that is what FB Prophet wants.
        horizon_days = int(self.forecast_period*time_interval_days[self.time_interval])
        #initial_days = total_days - NFOLDS * horizon_days
        initial_days = min(int(0.5*total_days), int(3*horizon_days)) ## this is recommended by FB Prophet
        #period_days = horizon_days
        period_days = min(int(0.2*initial_days), int(0.5*horizon_days)) # as recommended by FB Prophet

        if self.verbose >= 3:
            print("FB Prophet Cross-validation assumptions in days:")
            print(f"    Total number of days in train data = {total_days}")
            print(f"    Initial period of training days = {initial_days}")
            print(f"    Moving window Period of additional training days = {period_days}")
            print(f"    Forecast Horizon in days after every training = {horizon_days}")

        OFFSET = 0  # 5 days  # adjusting some days to take into account uneven months.
        #initial = str(initial_days-OFFSET) + " D"
        #period = str(period_days) + " D"
        #horizon = str(horizon_days+OFFSET) + " D"

        #### This is the simplest way to set these defaults to create a sliding window
        initial = int(dft.shape[0]/2) #*time_interval_days[timeinterval]
        horizon = self.forecast_period #*time_interval_days[timeinterval]
        period = max(2, int(self.forecast_period/2)) #*time_interval_days[timeinterval]
        if self.verbose >= 1:
            print("FB Prophet Cross-validation assumptions:")
            print(f"    OFFSET: {OFFSET}")
            print(f"    Initial period of training = {initial} {self.time_interval}")
            print(f"    Moving window Period of additional training: {period} {self.time_interval}")
            print(f"    Forecast Horizon after every training: {horizon} {self.time_interval}")

        # First  Fold -->
        #   Train Set: 0:initial
        #   Test Set: initial:(initial+horizon)
        # Second Fold -->
        #   Train Set: (period):(initial+period)
        #   Test Set: (initial+period):(initial+horizon+ period)
        # Format: '850 D'

        print("  Starting Prophet Cross Validation")
        with SuppressStdoutStderr():
            actuals, predictions, rmse_folds, norm_rmse_folds = easy_cross_validation(dft,actual,
                                        initial=initial, period=period,
                                          horizon=horizon)
            #df_cv = cross_validation(self.model, initial=initial, period=period,
            #                horizon=horizon)
            forecast_df_folds = copy.deepcopy(predictions)
        print("  End of Prophet Cross Validation")

        if self.verbose >= 1:
            print("Prophet CV DataFrame")
            #print(performance_metrics(df_cv).head())
        if self.verbose >= 2:
            print("Prophet plotting CV Metrics")
            #_ = plot_cross_validation_metric(df_cv, metric=self.scoring)
            #plt.show()

        #num_obs_folds = df_cv.groupby('cutoff')['ds'].count()

        # https://stackoverflow.com/questions/54405704/check-if-all-values-in-dataframe-column-are-the-same
        #a = num_obs_folds.to_numpy()
        #all_equal = (a[0] == a).all()

        #if not all_equal:
            #print("WARNING: All folds did not have the same number of observations in the validation sets.")
            #print("Num Test Obs Per fold")
            #print(num_obs_folds)

        #rmse_folds = []
        #norm_rmse_folds = []
        #forecast_df_folds = []

        #df_cv_grouped = df_cv.groupby('cutoff')
        #for (_, loop_df) in df_cv_grouped:
        #    rmse, norm_rmse = print_dynamic_rmse(loop_df['y'], loop_df['yhat'], dft['y'])
        #    rmse_folds.append(rmse)
        #    norm_rmse_folds.append(norm_rmse)
        #    forecast_df_folds.append(loop_df)

        # print(f"RMSE Folds: {rmse_folds}")
        # print(f"Norm RMSE Folds: {norm_rmse_folds}")
        # print(f"Forecast DF folds: {forecast_df_folds}")

        # forecast = self.predict(simple=False, return_train_preds=True)

        # ####  We are going to plot Prophet's forecasts differently since it is better
        # dfa = plot_prophet(dft, forecast);
        # # Prophet makes Incredible Predictions Charts!
        # ###  There can't be anything simpler than this to make Forecasts!
        # #self.model.plot(forecast);  # make sure to add semi-colon in the end to avoid plotting twice
        # # Also their Trend, Seasonality Charts are Spot On!
        # try:
        #     self.model.plot_components(forecast)
        # except:
        #     print('Error in FB Prophet components forecast. Continuing...')

        #rmse, norm_rmse = print_dynamic_rmse(dfa['y'], dfa['yhat'], dfa['y'])
        print('---------------------------')
        print('Final Prophet CV results:')
        print('---------------------------')
        rmse, norm_rmse = print_dynamic_rmse(actuals, predictions, actuals)

        print('Time taken (in seconds): %d' %((time.time()-start_time)))
        #return self.model, forecast, rmse, norm_rmse
        return self.model, forecast_df_folds, rmse_folds, norm_rmse_folds

    def refit(self, ts_df: pd.DataFrame) -> object:
        """
        Refits an already trained model using a new dataset
        Useful when fitting to the full data after testing with cross validation
        :param ts_df The time series data to be used for fitting the model
        :type ts_df pd.DataFrame
        :rtype object
        """

    def predict(
            self,
            testdata: Optional[pd.DataFrame] = None,
            forecast_period: Optional[int] = None,
            simple: bool = False,
            return_train_preds: bool = False) -> Optional[NDFrame]:
        """
        Return the predictions
        :param testdata The test dataframe containing the exogenous variables to be used for prediction.
        :type testdata Optional[pd.DataFrame]
        :param forecast_period The number of periods to make a prediction for.
        :type forecast_period Optional[int]
        :param simple If True, this method just returns the predictions.
        If False, it will return the standard error, lower and upper confidence interval (if available)
        :type simple bool
        :param return_train_preds If True, this method just returns the train predictions along with test predictions.
        If False, it will return only test predictions
        :type return_train_preds bool
        :rtype NDFrame
        """

        """
        Return the predictions
        # TODO: What about future exogenous variables?
        # https://towardsdatascience.com/forecast-model-tuning-with-additional-regressors-in-prophet-ffcbf1777dda
        """

        # if testdata is not None:
        #     warnings.warn(
        #         "Multivariate models are not supported by the AutoML prophet module." +
        #         "Univariate predictions will be returned for now."
        #     )

        # Prophet is a Little Complicated - You need 2 steps to Forecast
        ## 1. You need to create a dataframe to hold the predictions which specifies datetime
        ##    periods that you want to predict. It automatically creates one with both past
        ##    and future dates.
        ## 2. You need to ask Prophet to make predictions for the past and future dates in
        ##    that dataframe above.
        ## So if you had 2905 rows of data, and ask Prophet to predict for 365 periods,
        ##    it will give you predictions of the past (2905) and an additional 365 rows
        ##    of future (total: 3270) rows of data.
        ### This is where we take the first steps to make a forecast using Prophet:
        ##   1. Create a dataframe with datetime index of past and future dates

        # Next we ask Prophet to make predictions for those dates in the dataframe along with prediction intervals

        time_int = self.get_prophet_time_interval(for_cv=False)

        if self.univariate:
            if isinstance(testdata, int):
                forecast_period = testdata
            elif isinstance(testdata, pd.DataFrame):
                forecast_period = testdata.shape[0]
                if testdata.shape[0] != self.forecast_period:
                    self.forecast_period = testdata.shape[0]
            else:
                forecast_period = self.forecast_period
            future = self.model.make_future_dataframe(periods=self.forecast_period, freq=time_int)
        else:
            if isinstance(testdata, int) or testdata is None:
                print("(Error): Model is Multivariate, hence test dataframe must be provided for prediction.")
                return None
            elif isinstance(testdata, pd.DataFrame):
                forecast_period = testdata.shape[0]
                if testdata.shape[0] != self.forecast_period:
                    self.forecast_period = testdata.shape[0]
                future = self.prep_col_names_for_prophet(ts_df=testdata, test=True)
        print('Building Forecast dataframe. Forecast Period = %d' % self.forecast_period)
        ### This will work in both univariate and multi-variate cases now ######

        forecast = self.model.predict(future)

        # Return values for the forecast period only
        if simple:
            if return_train_preds:
                forecast = forecast['yhat']
            else:
                if forecast_period is None:
                    forecast = forecast['yhat']
                else:
                    forecast = forecast.iloc[-forecast_period:]['yhat']

        else:
            if return_train_preds:
                forecast = forecast
            else:
                if forecast_period is None:
                    forecast = forecast['yhat']
                else:
                    forecast = forecast.iloc[-forecast_period:]

        return forecast

    # TODO: Update: This method will not be used in CV since it is in D always.
    # Hence Remove the 'for_cv' argument
    def get_prophet_time_interval(self, for_cv: bool = False) -> str:
        """
        Returns the time interval in Prophet compatible format

        :param for_cv If False, this will return the format needed to make future dataframe (for univariate analysis)
        If True, this will return the format needed to be passed to the cross-validation object
        """
        if self.time_interval in ['months', 'month', 'm']:
            if for_cv:
                time_int = 'M'
            else:
                time_int = 'M'
        elif self.time_interval in ['days', 'daily', 'd']:
            if for_cv:
                time_int = 'days'
            else:
                time_int = 'D'
        elif self.time_interval in ['weeks', 'weekly', 'w']:
            if for_cv:
                time_int = 'W'
            else:
                time_int = 'W'
        # TODO: Add time_int for other options if they are different for CV and for future forecasts
        elif self.time_interval in ['qtr', 'quarter', 'q']:
            time_int = 'Q'
        elif self.time_interval in ['years', 'year', 'annual', 'y', 'a']:
            time_int = 'Y'
        elif self.time_interval in ['hours', 'hourly', 'h']:
            time_int = 'H'
        elif self.time_interval in ['minutes', 'minute', 'min', 'n']:
            time_int = 'M'
        elif self.time_interval in ['seconds', 'second', 'sec', 's']:
            time_int = 'S'
        else:
            time_int = 'W'

        return time_int

    def prep_col_names_for_prophet(self, ts_df: pd.DataFrame, test: bool = False) -> pd.DataFrame:
        """
        Renames the columns of the input dataframe to the right format needed by Prophet
        Target is renamed to 'y' and the time column is renamed to 'ds'
        # TODO: Complete docstring
        """

        if self.time_col not in ts_df.columns:
            #### This happens when time_col is not found but it's actually the index. In that case, reset index
            data = ts_df.reset_index()
        else:
            data = ts_df.copy(deep=True)

        if self.time_col not in data.columns:
            print("(Error): You have not provided the time_column values. This will result in an error")

        if test is False:
            data = data.rename(columns={self.time_col: 'ds', self.original_target_col: 'y'})
        else:
            data = data.rename(columns={self.time_col: 'ds'})

        return data

def plot_prophet(dft, forecastdf):
    """
    This is a different way of plotting Prophet charts as described in the following article:
    Source: https://nextjournal.com/viebel/forecasting-time-series-data-with-prophet
    Reproduced with gratitude to the author.
    """
    dft = copy.deepcopy(dft)
    forecastdf = copy.deepcopy(forecastdf)
    dft.set_index('ds', inplace=True)
    forecastdf.set_index('ds', inplace=True)
    dft.index = pd.to_datetime(dft.index)
    connect_date = dft.index[-2]
    mask = (forecastdf.index > connect_date)
    predict_df = forecastdf.loc[mask]
    viz_df = dft.join(predict_df[['yhat', 'yhat_lower', 'yhat_upper']], how='outer')
    _, ax1 = plt.subplots(figsize=(20, 10))
    ax1.plot(viz_df['y'], color='red')
    ax1.plot(viz_df['yhat'], color='green')
    ax1.fill_between(viz_df.index, viz_df['yhat_lower'], viz_df['yhat_upper'],
                     alpha=0.2, color="darkgreen")
    ax1.set_title('Actual (Red) vs Forecast (Green)')
    ax1.set_ylabel('Values')
    ax1.set_xlabel('Date Time')
    plt.show(block=False)
    return viz_df
#################################
from sklearn.metrics import mean_squared_error
from fbprophet import Prophet
import time
import pdb
import copy
import matplotlib.pyplot as plt
def easy_cross_validation(train, target, initial, horizon, period):
    n_folds = int(((train.shape[0]-initial)/period)-1)
    y_preds = pd.DataFrame()
    print('Max. iterations using sliding window cross validation = %d' %n_folds)
    start_time = time.time()
    start_p = 0  ## this represents start of train fold
    end_p = initial  ## this represents end of train fold
    start_s = initial  ## this represents start of test fold
    end_s = initial + horizon ### this represents end of test fold
    rmse_means = []
    norm_rmse_means = []
    y_trues = pd.DataFrame()
    for i in range(n_folds):
        #start_p += i*period
        end_p += i*period
        train_fold = train[start_p:end_p]
        start_s += i*period
        end_s += i*period
        test_fold = train[start_s: end_s]
        if len(test_fold) == 0:
            break
        model = Prophet(growth="linear")
        kwargs = {'iter':1e2} ## this limits iterations and hence speeds up prophet
        model.fit(train_fold, **kwargs)
        future_period = model.make_future_dataframe(freq="MS",periods=horizon)
        forecast_df = model.predict(future_period)
        y_pred = forecast_df.iloc[start_s:end_s]['yhat']
        if i == 0:
            y_preds = copy.deepcopy(y_pred)
        else:
            y_preds = y_preds.append(y_pred)
        rmse_fold, rmse_norm = print_dynamic_rmse(test_fold[target],y_pred,test_fold[target])
        print('Cross Validation window: %d completed' %(i+1,))
        rmse_means.append(rmse_fold)
        norm_rmse_means.append(rmse_norm)
    ### This is where you consolidate the CV results ####
    #print('Time Taken = %0.0f mins' %((time.time()-start_time)/60))
    rmse_mean = np.mean(rmse_means)
    #print('Average CV RMSE over %d windows (macro) = %0.5f' %(i,rmse_mean))
    y_trues = train[-y_preds.shape[0]:][target]
    cv_micro = np.sqrt(mean_squared_error(y_trues.values,
                                          y_preds.values))
    #print('Average CV RMSE of all predictions (micro) = %0.5f' %cv_micro)
    try:
        quick_ts_plot(train[target], y_preds[-horizon:])
    except:
        print('Error: Not able to plot Prophet CV results')
    return y_trues, y_preds, rmse_means, norm_rmse_means
##################################################################################
