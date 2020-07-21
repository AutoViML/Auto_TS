from typing import Optional
import warnings
import numpy as np  # type: ignore
import pandas as pd # type: ignore
import copy
import matplotlib.pyplot as plt # type: ignore

# helper functions
from ..utils import print_dynamic_rmse

from fbprophet import Prophet # type: ignore
from fbprophet.diagnostics import cross_validation

from .build_base import BuildBase

#### Suppress INFO messages from FB Prophet!
import logging
logging.getLogger('fbprophet').setLevel(logging.WARNING)
import pdb

class BuildProphet(BuildBase):
    def __init__(self, forecast_period, time_interval, 
        scoring, verbose, conf_int):
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
        self.model = Prophet(interval_width=self.conf_int)

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

        ts_df = copy.deepcopy(ts_df)

        ##### if you are going to use matplotlib with prophet data, it gives an error unless you do this.
        pd.plotting.register_matplotlib_converters()
        
        #### You have to import Prophet if you are going to build a Prophet model #############
        try:
            print('Preparing Time Series data for FB Prophet: sample row before\n', ts_df[time_col].head(1))
            df = ts_df.rename(columns={time_col: 'ds', target_col: 'y'})
            print('Time Series data: sample row after transformation\n', df.head(1))
        except:
            #### This happens when time_col is not found but it's actually the index. In that case, reset index
            print('Preparing Time Series data for FB Prophet: sample row before\n', ts_df.head(1))
            df = ts_df.reset_index()
            df = df.rename(columns={time_col: 'ds', target_col: 'y'})
            print('Time Series data: sample row after transformation\n', df.head(1))
        actual = 'y'
        timecol = 'ds'
        dft = df[[timecol, actual]]
        
        ##### For most Financial time series data, 80% conf interval is enough...
        print('    Fit-Predict data (shape=%s) with Confidence Interval = %0.2f...' % (dft.shape, self.conf_int))
        ### Make Sure you lower your desired interval width from the normal 95% to a more realistic 80%
        
        #### TODO: Start from Monday evening (7/20)
        # Add seasonality components to Prophet call (in init)
        # Add regressors
        # Only then train

        # m = Prophet(yearly_seasonality=True
        #     ,weekly_seasonality=True
        #     ,daily_seasonality=False
        #     #,seasonality_mode='multiplicative'
        #     ,seasonality_prior_scale=25
        #     ,changepoint_range=0.95
        #    )
        # m.add_regressor('0')
        # m.add_regressor('1')
        # m.add_regressor('2')
        # m.fit(train)

        ####


        self.model.fit(dft)

        num_obs = dft.shape[0]
        NFOLDS = self.get_num_folds_from_cv(cv)
        window_size = int(num_obs/NFOLDS)
        
        time_int_cv = self.get_prophet_time_interval(for_cv=True)
        initial = str(window_size - self.forecast_period) + " " + time_int_cv
        period = str(window_size) + " " + time_int_cv
        horizon = self.forecast_period

        print("Prophet CV Diagnostics:")
        print(f"NumObs: {num_obs}")
        print(f"NFOLDS: {NFOLDS}")
        print(f"window_size: {window_size}")
        print(f"initial: {initial}")
        print(f"period: {period}")
        print(f"horizon: {horizon}")


        # df_cv = cross_validation(
        #     self.model,
        #     initial=initial,  # '65 W', #M for months
        #     period=period,  # '26 W',
        #     horizon=horizon #'52 W'
        # ) 
        
        # # first: train: 0 to 64 Test 65 to 65+52
        # # second: train: 0+26 to 65+26 Test 65+26 to 65+26+52
        # # next: train: 0+26+26. to 65+26+26. Test 65+26+26.. to 65+26+26+52
        
        # print("Prophet CV DataFrame")
        # print(df_cv)

        # print("Prophet Num Obs Per fold")
        # print(df_cv.groupby('cutoff')['ds'].count())
        

        forecast = self.predict(simple=False, return_train_preds=True)

        ####  We are going to plot Prophet's forecasts differently since it is better
        dfa = plot_prophet(dft, forecast);
        # Prophet makes Incredible Predictions Charts!
        ###  There can't be anything simpler than this to make Forecasts!
        #self.model.plot(forecast);  # make sure to add semi-colon in the end to avoid plotting twice
        # Also their Trend, Seasonality Charts are Spot On!
        try:
            self.model.plot_components(forecast)
        except:
            print('Error in FB Prophet components forecast. Continuing...')
        
        rmse, norm_rmse = print_dynamic_rmse(dfa['y'], dfa['yhat'], dfa['y'])
        return self.model, forecast, rmse, norm_rmse

    def refit(self, ts_df: pd.DataFrame) -> object:
        """
        Refits an already trained model using a new dataset
        Useful when fitting to the full data after testing with cross validation
        :param ts_df The time series data to be used for fitting the model
        :type ts_df pd.DataFrame
        :rtype object
        """

    #  def predict(
    #     self,
    #     X_exogen: Optional[pd.DataFrame]=None,
    #     forecast_period: Optional[int] = None,
    #     simple: bool = True) -> NDFrame:
    #     """
    #     Return the predictions
    #     :param X_exogen The test dataframe containing the exogenous varaiables to be used for predicton.
    #     :type X_exogen Optional[pd.DataFrame]
    #     :param forecast_period The number of periods to make a prediction for.
    #     :type forecast_period Optional[int]
    #     :param simple If True, this method just returns the predictions. 
    #     If False, it will return the standard error, lower and upper confidence interval (if available)
    #     :type simple bool
    #     :rtype NDFrame
    #     """


    def predict(
        self,
        X_exogen: Optional[pd.DataFrame]=None,
        forecast_period: Optional[int] = None,
        simple: bool = True,
        return_train_preds: bool = False
        ):
        """
        Return the predictions
        # TODO: What about future exogenous variables?
        # https://towardsdatascience.com/forecast-model-tuning-with-additional-regressors-in-prophet-ffcbf1777dda
        """

        if X_exogen is not None:
            warnings.warn(
                "Multivariate models are not supported by the AutoML prophet module." +  
                "Univariate predictions will be returned for now."                
            )

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
        print('Building Forecast dataframe. Forecast Period = %d' % self.forecast_period)
        # Next we ask Prophet to make predictions for those dates in the dataframe along with predn intervals
        if self.time_interval in ['months', 'month', 'm']:
            time_int = 'M'
        elif self.time_interval in ['days', 'daily', 'd']:
            time_int = 'D'
        elif self.time_interval in ['weeks', 'weekly', 'w']:
            time_int = 'W'
            # seasonal_period = 52  # TODO: #7 Unused - Check if this is needed or somethig is missing
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
            # time_interval = 'S'  # TODO: I think this should be time_int
        else:
            time_int = 'W'

        if forecast_period is None:
            forecast_period = self.forecast_period
        
        future = self.model.make_future_dataframe(periods=forecast_period, freq=time_int)
        forecast = self.model.predict(future)

        # Return values for the forecast period only
        if simple:
            if return_train_preds:
                forecast = forecast['yhat']
            else:
                forecast = forecast.iloc[-forecast_period:]['yhat']
            
        else:
            if return_train_preds:
                forecast = forecast
            else:
                forecast = forecast.iloc[-forecast_period:]
            
        return forecast

    def get_prophet_time_interval(self, for_cv: bool =False) -> str:
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
    viz_df = dft.join(predict_df[['yhat', 'yhat_lower', 'yhat_upper']],
                      how='outer')
    _ , ax1 = plt.subplots(figsize=(20, 10))
    ax1.plot(viz_df['y'], color='red')
    ax1.plot(viz_df['yhat'], color='green')
    ax1.fill_between(viz_df.index, viz_df['yhat_lower'], viz_df['yhat_upper'],
                     alpha=0.2, color="darkgreen")
    ax1.set_title('Actuals (Red) vs Forecast (Green)')
    ax1.set_ylabel('Values')
    ax1.set_xlabel('Date Time')
    plt.show(block=False)
    return viz_df

