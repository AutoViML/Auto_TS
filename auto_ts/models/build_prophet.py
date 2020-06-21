from typing import Optional
import warnings
import numpy as np  # type: ignore
import pandas as pd # type: ignore
import copy
import matplotlib.pyplot as plt # type: ignore
# helper functions
from ..utils import print_dynamic_rmse
# imported Prophet from fbprophet pkg
from fbprophet import Prophet # type: ignore
#### Suppress INFO messages from FB Prophet!
import logging
logging.getLogger('fbprophet').setLevel(logging.WARNING)
import pdb

class BuildProphet():
    def __init__(self, forecast_period, time_interval, 
        score_type, verbose, conf_int):
        """
        Automatically build a Prophet Model
        """
        self.forecast_period = forecast_period
        self.time_interval = time_interval
        self.score_type = score_type  # TODO: Unused in class
        self.verbose = verbose # TODO: Unused in class
        self.conf_int = conf_int
        self.model = Prophet(interval_width=self.conf_int)

    def fit(self, ts_df, time_col, target):
        """
        Build a Time Series Model using Facebook Prophet which is a powerful model.
        """
        ts_df = copy.deepcopy(ts_df)
        #df.rename(columns={time_col:'ds',target:'y'},inplace=True)
        ##### if you are going to use matplotlib with prophet data, it gives an error unless you do this.
        pd.plotting.register_matplotlib_converters()
        #### You have to import Prophet if you are going to build a Prophet model #############
        try:
            print('Preparing Time Series data for FB Prophet: sample row before\n', ts_df[time_col].head(1))
            df = ts_df.rename(columns={time_col: 'ds', target: 'y'})
            print('Time Series data: sample row after transformation\n', df.head(1))
        except:
            #### THis happens when time_col is not found but it's actually the index. In that case, reset index
            print('Preparing Time Series data for FB Prophet: sample row before\n', ts_df.head(1))
            df = ts_df.reset_index()
            df = df.rename(columns={time_col: 'ds', target: 'y'})
            print('Time Series data: sample row after transformation\n', df.head(1))
        actual = 'y'
        timecol = 'ds'
        dft = df[[timecol, actual]]
        ##### For most Financial time series data, 80% conf interval is enough...
        print('    Fit-Predict data (shape=%s) with Confidence Interval = %0.2f...' % (dft.shape, self.conf_int))
        ### Make Sure you lower your desired interval width from the normal 95% to a more realistic 80%
        # model = Prophet(interval_width=self.conf_int) # moved to init
        self.model.fit(dft)
        forecast = self.predict(simple=False, return_train_preds=True)

        # act_n = len(dft)  # TODO: Not used anywhere, hence commenting
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
        #submit = dfplot[-self.forecast_period:]
        #submit.drop('Actuals',axis=1,inplace=True)
        #submit.rename(columns={'yhat':target},inplace=True)
        #print('Forecast Data frame size %s ready to submit' %(submit.shape,))
        return self.model, forecast, rmse, norm_rmse

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

