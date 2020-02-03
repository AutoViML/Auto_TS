import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
# imported Prophet from fbprophet pkg
from fbprophet import Prophet
# helper functions
from ..utils import print_dynamic_rmse


def build_prophet_model(ts_df, time_col, target, forecast_period, score_type,
                        verbose, conf_int):
    """
    Build a Time Series Model using Facebook Prophet which is a powerful model.
    """
    ##### if you are going to use matplotlib with prophet data, it gives an error unless you do this.
    pd.plotting.register_matplotlib_converters()
    #### You have to import Prophet if you are going to build a Prophet model #############
    df = ts_df[:]
    try:
        df[time_col].head()
    except:
        #### THis happens when time_col is not found but it's actually the index. In that case, reset index
        df = ts_df.reset_index()
    df.rename(columns={time_col: 'ds', target: 'y'}, inplace=True)
    actual = 'y'
    timecol = 'ds'
    dft = df[[timecol, actual]]
    ##### For most Financial time series data, 80% conf interval is enough...
    print('    Fit-Predict data (shape=%s) with Confidence Interval = %0.2f...' % (dft.shape, conf_int))
    ### Make Sure you lower your desired interval width from the normal 95% to a more realistic 80%
    model = Prophet(interval_width=conf_int)
    model.fit(dft)
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
    print('Building Forecast dataframe. Forecast Period = %d' % forecast_period)
    # Next we ask Prophet to make predictions for those dates in the dataframe along with predn intervals
    future = model.make_future_dataframe(periods=forecast_period)
    forecast = model.predict(future)
    act_n = len(dft)
    ####  We are going to plot Prophet's forecasts differently since it is better
    dfa = plot_prophet(dft, forecast)
    # Prophet makes Incredible Predictions Charts!
    ###  There can't be anything simpler than this to make Forecasts!
    #model.plot(forecast);  # make sure to add semi-colon in the end to avoid plotting twice
    # Also their Trend, Seasonality Charts are Spot On!
    try:
        model.plot_components(forecast);
    except:
        print('Error in FB Prophet components forecast. Continuing...')
    rmse, norm_rmse = print_dynamic_rmse(dfa['y'], dfa['yhat'], dfa['y'])
    #submit = dfplot[-forecast_period:]
    #submit.drop('Actuals',axis=1,inplace=True)
    #submit.rename(columns={'yhat':target},inplace=True)
    #print('Forecast Data frame size %s ready to submit' %(submit.shape,))
    return model, forecast, rmse, norm_rmse


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
    fig,ax1 = plt.subplots(figsize=(20, 10))
    ax1.plot(viz_df['y'], color='red')
    ax1.plot(viz_df['yhat'], color='green')
    ax1.fill_between(viz_df.index, viz_df['yhat_lower'], viz_df['yhat_upper'],
                     alpha=0.2, color="darkgreen")
    ax1.set_title('Actuals (Red) vs Forecast (Green)')
    ax1.set_ylabel('Values')
    ax1.set_xlabel('Date Time')
    plt.show();
    return viz_df


# def print_dynamic_rmse(actuals, predicted, original):
#     """
#     This utility calculates rmse between actuals and predicted. However, it does one more.
#     Since in dynamic forecast, we need the longer original, it calculates Normalized RMSE
#     using the original array's std deviation. That way, the forecast of 2 values does not
#     result in a larger Normalized RMSE since the std deviation of 2 values will be v small.
#     """
#     rmse = np.sqrt(np.mean((actuals - predicted)**2))
#     norm_rmse = rmse/original.std()
#     print('    RMSE = {:,.2f}'.format(rmse))
#     print('    Std Deviation of Originals = {:,.2f}'.format(original.std()))
#     print('    Normalized RMSE = %0.0f%%' %(100*norm_rmse))
#     return rmse, norm_rmse
