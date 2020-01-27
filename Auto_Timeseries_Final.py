####################################################################################
####                     Auto Time Series Final  0.0.1                          ####
####                           Python 3 Version                                 ####
####                      Developed by Ram Seshadri                             ####
####                        All Rights Reserved                                 ####
####################################################################################
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings. filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
sns.set(style="white", color_codes=True)
import copy
import pdb
from itertools import cycle
from collections import defaultdict, Counter
import operator
from scipy import interp
import time
import itertools
#######################################
# TSA from Statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
###########################################
class colorful:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'
#################################################
def top_correlation_to_name(stocks, column_name, searchstring, top=5):
    """
    ####################################################################################
    This function draws a correlation chart of the top "x" rows of a data frame that are highly
    correlated to a selected row in the dataframe. You can think of the rows of the input
    dataframe as containing stock prices or fund flows or product sales and the columns should
    contain time series data of prices or flows or sales over multiple time periods.
    Now this program will allow you to select the top 5 or 10 rows that are highly correlated
    to a given row selected by the column: column_name and using a search string "searchstring".
    The program will search for the search string in that column column_name and return a list
    of 5 or 10 rows that are the most correlated to that selected row. If you give "top" as
    a float ratio then it will use the ratio as the cut off point in the correlation
    coefficient to select rows.
    ####################################################################################
    """
    #### First increment top by 1 since you are asking for top X names in addition to the one you have, top += 1
    incl = [x for x in list(stocks) if x not in column_name]
    ### First drop all NA rows since they will mess up your correlations, stocks.dropna(inplace=True)
    if stocks.empty:
        print('After dropping NaNs, the data frame has become empty.')
        return
    ### Now find the highest correlated rows to the selected row ###
    try:
        index_val = search_string(stocks, column_name,searchstring).index[0]
    except:
        print('Not able to find the search string in the column.')
        return
    ### Bring that selected Row to the top of the Data Frame
    df = stocks[:]
    df["new"] = range(l, len(df)+l)
    df.loc[index_val,"new"] = 0
    stocks = df.sort_values("new").drop("new",axis=1)
    stocks.reset_index(inplace=True,drop=True)
    ##### Now calculate the correlation coefficients of other rows with the Top row
    try:
        cordf = pd.DataFrame(stocks[incl].T.corr().sort_values(0,ascending=False))
    except:
        print('Cannot calculate Correlations since Dataframe contains string values or objects.')
        return
    try:
        cordf = stocks[column_name].join(cordf)
    except:
        cordf = pd.concat((stocks[column_name],cordf),axis=1)
    #### Visualizing the top 5 or 10 or whatever cut-off they have given for Corr Coeff
    if top >= 1:
        top10index = cordf.sort_values(0,ascending=False).iloc[:top,:3].index
        top10names = cordf.sort_values(0,ascending=False).iloc[:top,:3][column_name]
        top10values = cordf.sort_values(0,ascending=False)[0].values[:top]
    else:
        top10index = cordf.sort_values(0,ascending=False)[
                        cordf.sort_values(0,ascending=False)[0].values>=top].index
        top10names = cordf.sort_values(0,ascending=False)[cordf.sort_values(
                            0,ascending=False)[0].values>=top][column_name]
        top10alues = cordf.sort_values(0,ascending=False)[cordf.sort_values(
                                0,ascending=False)[0].values>=top][0]
    print(top10names,top10values)
    #### Now plot the top rows that are highly correlated based on condition above
    stocksloc = stocks.iloc[top10index]
    #### Visualizing using Matplotlib ###
    stocksloc = stocksloc.T
    stocksloc = stocksloc.reset_index(drop=True)
    stocksloc.columns = stocksloc.iloc[0].values.tolist()
    stocksloc.drop(0).plot(subplots=True, figsize=(15,10),legend=False,
                title="Top %s Correlations to %s" %(top,searchstring))
    [ax.legend(loc=1) for ax in plt.gcf().axes]
    plt.tight_layout()
    plt.show()
###################################################################################
import matplotlib.dates as mdates
import numpy as np
def using_where(x):
    return np.where(x == 1,'g','r')
from itertools import cycle, combinations
def time_series_plot(y, lags=31, title='Original Time Series', chart_type='line',
                                            chart_time='years'):
    """
    Plot a Time Series along with how it will look after differencing and what its
    AR/MA lags will be by viewing the ACF and PACF, along with its histogram.
    You just need to provide the time series (y) as a Series. Index is assumed
    to be Pandas datetime. It assumes that you want to see default lags of 31.
    But you can modify it to suit.
    """
    colors = cycle('byrcmgkbyrcmgkbyrcmgkbyrcmgkbyr')
    fig = plt.figure(figsize=(20, 20))
    grid = plt.GridSpec(3, 2, wspace=0.5, hspace=0.5)
    fig.subplots_adjust(hspace=1)
    ########## Use the gridspec function ##############
    ts_ax = plt.subplot(grid[0, 0:])
    diff_ax = plt.subplot(grid[1, 0])
    hist_ax = plt.subplot(grid[1, 1])
    acf_ax = plt.subplot(grid[2, 0])
    pacf_ax = plt.subplot(grid [2,1])
    ### Draw multiple kinds of graphs here to each subplot axis ###
    if chart_type == 'line':
        y.plot(ax=ts_ax,color=next(colors))
    else:
        if chart_time == 'years':
            majors = mdates.YearLocator() # every year
            minors = mdates.MonthLocator() # every month
            majorsFmt = mdates.DateFormatter('%Y')
        elif chart_time == 'months':
            majors = mdates.YearLocator() # every year
            minors = mdates.MonthLocator() # every month
            majorsFmt = mdates.DateFormatter('\n\n\n%b\n%Y')
        elif chart_time == 'weeks':
            majors = mdates.MonthLocator()
            minors = mdates.WeekdayLocator(byweekday=(1), interval=1)
            majorsFmt = mdates.DateFormatter('\n\n\n%b\n%Y')
        elif chart_time == 'days':
            majors = mdates.DayLocator(bymonthday=None, interval=1, tz=None)
            minors = mdates.HourLocator(byhour=None, interval=1, tz=None)
            majorsFmt = mdates.DateFormatter('\n\n\n%d\n%b')
        else:
            majors = mdates.YearLocator() # every year
            minors = mdates.MonthLocator() # every month
            majorsFmt = mdates.DateFormatter('\n\n\n%b\n%Y')
        try:
            #### this works in most cases but in some cases, it gives an error
            ts_ax.bar(y.index,height=y,width=20,color=list((y>0).astype(int).map({1:'g',0:'r'}).values))
        except:
            #### In some cases where y is a dataframe, this might work.
            yindex = y.index
            yvalues = y.values.ravel()
            ts_ax.bar(yindex,height=yvalues,width=20,color=list(using_where((yvalues>0).astype(int)).ravel()))
        ts_ax.xaxis.set_major_locator(majors)
        ts_ax.xaxis.set_major_formatter(majorsFmt)
        ts_ax.xaxis.set_minor_locator(minors)
        ts_ax.format_xdata = mdates.DateFormatter('%Y-%m-%d')
        ts_ax.grid(True)
    ts_ax.set_title(title)
    y.diff(1).plot(ax=diff_ax, color=next(colors))
    diff_ax.set_title('After Differencing = 1')
    y.plot(ax=hist_ax, kind='hist', bins=25,color=next(colors))
    hist_ax.set_title('Histogram for Original Series')
    try:
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
        acf_ax.set_title('ACF for Original Series')
    except:
        acf_ax.set_title('Data Error: Could not draw ACF for Original Series')
    try:
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
        pacf_ax.set_title('PACF for Original Series')
    except:
        pacf_ax.set_title('Data Error: Could not draw PACF for Original Series')
    [ax.set_xlim(0) for ax in [acf_ax, pacf_ax]]
    plt.show()
##################################################################
def test_stationarity(timeseries,maxlag=2, regression='c', autolag=None, window=None, plot=False, verbose=False):
    """
    Check unit root stationarity of a time series array or an entire dataframe.
    Note that you must send in a dataframe as df.values.ravel() - otherwise ERROR.
    Null hypothesis: the series is non-stationary.
    If p >= alpha, the series is non-stationary.
    If p < alpha, reject the null hypothesis (has unit root stationarity).
    Original source: http://www.analyticsvidhya.com/blog/2016/02/time-series-forecasting-codes-python/
    Function: http://statsmodels.sourceforge.net/devel/generated/statsmodels.tsa.stattools.adfuller.html
    window argument is only required for plotting rolling functions. Default=4.
    """
    # set defaults (from function page)
    if type(timeseries) == pd.DataFrame:
        print('modifying time series dataframe into an array to test')
        timeseries = timeseries.values.ravel()
    if regression is None:
        regression = 'c'
    if verbose:
        print('Running Augmented Dickey-Fuller test with paramters:')
        print('maxlag: {}'.format(maxlag))
        print('regression: {}'.format(regression))
        print('autolag: {}'.format(autolag))
    alpha =0.05
    if plot:
        if window is None:
            window = 4
        #Determing rolling statistics
        rolmean = timeseries.rolling(window=window, center=False).mean()
        rolstd = timeseries.rolling(window=window, center=False).std()
        #Plot rolling statistics:
        orig = pit.plot(timeseries, color='blue', label='Original')
        mean = plt.plot(rolmean, color='red', label='Rolling Mean ({})'.format(window))
        std = plt.plot(rolstd, color='black', label='Rolling Std ({})'.format(window))
        plt.legend(loc='best')
        plt.title('Rolling Mean & Standard Deviation')
        plt.show(block=False)
    #Perform Augmented Dickey-Fuller test:
    try:
        dftest = smt.adfuller(timeseries, maxlag=maxlag, regression=regression, autolag=autolag)
        dfoutput = pd.Series(dftest[0:4], index=['Test Statistic',
                                                'p-value',
                                                '#Lags Used',
                                                'Number of Observations Used',
                                                ])
        for key,value in dftest[4].items():
            dfoutput['Critical Value (%s)' %key] = value
        if verbose:
            print('Results of Augmented Dickey-Fuller Test:')
            print(dfoutput)
        if dftest[1] >= alpha:
            print(' this series is non-stationary')
        else:
            print(' this series is stationary')
        return dfoutput
    except:
        print('Augment Dickey-Fuller test gives an error')
        return
########
def convert_timeseries_dataframe_to_supervised(df, namevars, target, n_in=1, n_out=0, dropT=True):
    """
    Transform a time series in dataframe format into a supervised learning dataset while
    keeping dataframe intact.
    Arguments:
        df: A timeseries dataframe that you want to convert to Supervised dataset.
        namevars: columns that you want to lag in the data frame. Other columns will be untouched.
        target: this is the target variable you intend to use in supervised learning
        n_in: Number of lag periods as input (X).
        n_out: Number of future periods (optional) as output for the taget variable (y).
        dropT: Boolean - whether or not to drop columns at time 't'.
        Returns:
        df: This is the transformed data frame with the time series columns laggged.
        Note that the original columns are dropped if you set the 'dropT' argument to True.
        If not, they are preserved.
    This Pandas DataFrame of lagged time series data is immediately available for supervised learning.
    """
    df = df[:]
    # Notice that we will create a sequence of columns from name vars with suffix (t-n,... t-1), etc.
    drops = []
    for i in range(n_in, -1, -1):
        if i == 0:
            for var in namevars:
                addname = var + '(t)'
                df.rename(columns={var:addname},inplace=True)
                drops.append(addname)
        else:
            for var in namevars:
                addname = var + '(t-' + str(i) +')'
                df[addname] = df[var].shift(i)
    ## forecast sequence (t, t+1,... t+n)
    if n_out == 0:
        n_out = False
    for i in range(1, n_out):
        for var in namevars:
            addname = var + '(t+' + str(i) + ')'
            df[addname] = df[var].shift(-i)
    #	drop rows with NaN values
    df.dropna(inplace=True,axis=0)
    #	put it all together
    target = target+'(t)'
    if dropT:
        ### If dropT is true, all the "t" series of the target column (in case it is in the namevars)
        ### will be removed if you don't want the target to learn from its "t" values.
        ### Similarly, we will also drop all the "t" series of name_vars if you set dropT to Trueself.
        try:
            drops.remove(target)
        except:
            pass
        df.drop(drops, axis=1, inplace=True)
    preds = [x for x in list(df) if x not in [target]]
    return df, target, preds
    ############
##### This function loads a time series data and sets the index as a time series
def load_ts_data(filename,ts_column,sep,target):
    """
    This function loads a given filename into a pandas dataframe and sets the
    ts_column as a Time Series index. Note that filename should contain the full
    path to the file.
    """
    if isinstance(filename,str):
        codes_list = ['utf-8','iso-8859-1','cp1252','latin1']
        print('First loading %s and then setting %s as date time index...' %(filename,ts_column))
        for codex in codes_list:
            try:
                df = pd.read_csv(filename, index_col=None, sep=sep, encoding=codex)
                df.index = pd.to_datetime(df.pop(ts_column))
                break
            except:
                print('    Encoder %s or Date time type not working for reading this file...' %codex)
                continue
    else:
        ### If filename is not a string, it must be a dataframe and can be loaded
        dft = copy.deepcopy(filename)
        preds = [x for x in list(dft) if x not in [target]]
        df = dft[[ts_column]+[target]+preds]
    return df
####################################################
def find_lowest_pq(df):
    """
    This is an auto-ARIMA function that iterates through parameters pdq and finds the best
    based on aan eval metric sent in as input.
    """
    dicti = {}
    for ma in list(df):
        try:
            dicti[ma +''+ df[ma].idxmin()]= df[ma].sort_values()[0]
        except:
            pass
    lowest_bic = min(dicti.items(), key=operator.itemgetter(1))[1]
    lowest_pq = min(dicti.items(), key=operator.itemgetter(1))[0]
    ma_q = int(lowest_pq.split(' ')[0][2:])
    ar_p = int(lowest_pq.split(' ')[1][2:])
    print('Best AR order p = %d, MA order q = %d, Interim metric = %0.3f' %(ar_p, ma_q, lowest_bic))
    return ar_p, ma_q, lowest_bic
###################################################################################
def find_best_pdq_or_PDQ(ts_train, metric, p_max, d_max, q_max,
                non_seasonal_pdq, seasonal_period, seasonality=False,verbose=0):
    p_min = 0
    d_min = 0
    q_min = 0
    if seasonality:
        ns_p = non_seasonal_pdq[0]
        ns_d = non_seasonal_pdq[1]
        ns_q = non_seasonal_pdq[2]
    # Initialize a DataFrame to store the results
    iteration = 0
    results_dict = {}
    for d_val in range(d_min,d_max+1):
        print('\nDifferencing = %d' %d_val)
        results_bic = pd.DataFrame(index=['AR{}'.format(i) for i in range(p_min,p_max+1)],
                        columns=['MA{}'.format(i) for i in range(q_min,q_max+1)])
        for p_val, q_val in itertools.product(range(p_min,p_max+1), range(q_min,q_max+1)):
            if p_val==0 and d_val==0 and q_val==0:
                results_bic.loc['AR{}'.format(p_val), 'MA{}'.format(q_val)] = np.nan
                continue
            try:
                if seasonality:
                    #### In order to get forecasts to be in the same value ranges of the
                    #### orig_endogs, you must set the simple_differencing = False and
                    #### the start_params to be the same as ARIMA.
                    #### THat is the only way to ensure that the output of this
                    #### model is comparable to other ARIMA models
                    model = sm.tsa.statespace.SARIMAX(ts_train, order=(ns_p,ns_d,ns_q),
                                        seasonal_order=(p_val,d_val,q_val,seasonal_period),
                                        enforce_stationarity=False,
                                        enforce_invertibility=False,
                                        simple_differencing=False, trend='ct',
                                        start_params=[0, 0, 0,1])
                else:
                    model = sm.tsa.SARIMAX(ts_train, order=(p_val, d_val, q_val),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False,
                                    simple_differencing=False, trend='ct',
                                    start_params=[0, 0, 0,1]
                                        )
                    results = model.fit()
                    results_bic.loc['AR{}'.format(p_val), 'MA{}'.format(q_val)] = eval('results.'+metric)
                    if iteration % 10 == 0:
                        print('    Iteration %d completed...' %iteration)
                        iteration += 1
                    elif iteration >= 100:
                        print('    Ending Iterations at %d' %iteration)
                        break
            except:
                iteration += 1
                continue
        results_bic = results_bic[results_bic. columns].astype(float)
        interim_d = d_val
        if results_bic.isnull().all().all():
            print('    D = %d results in an empty ARMA set. Setting Seasonality to False since model might overfit' %d_val)
            #### Set Seasonality to False if this empty condition happens repeatedly ####
            seasonality = False
            continue
        else:
            seasonality = True
        interim_p, interim_q, interim_bic = find_lowest_pq(results_bic)
        if verbose == 1:
            fig, ax = plt.subplots(figsize=(20,10))
            ax = sns.heatmap(results_bic, mask=results_bic.isnull(), ax=ax, annot=True, fmt='.0f')
            ax.set_title(metric)
        results_dict[str(interim_p)+' '+str(interim_d)+' '+str(interim_q)] = interim_bic
    try:
        best_bic = min(results_dict.items(), key=operator.itemgetter(1))[1]
        best_pdq = min(results_dict.items(), key=operator.itemgetter(1))[0]
        best_p = int(best_pdq.split(' ')[0])
        best_d = int(best_pdq.split(' ')[1])
        best_q = int(best_pdq.split(' ')[2])
    except:
        best_p = copy.deepcopy(p_val)
        best_q = copy.deepcopy(q_val)
        best_d = copy.deepcopy(d_val)
        best_bic = 0
    return best_p, best_d, best_q, best_bic, seasonality
####################################################################################
def build_best_SARIMAX_model(ts_df, metric, seasonality=False, seasonal_period=None,
                                    p_max=12, d_max=2, q_max=12,forecast_period=2,verbose=0):
    ############ Split the data set into train and test for Cross Validation Purposes ########
    ts_train = ts_df[:-forecast_period]
    ts_test = ts_df[-forecast_period:]
    if verbose == 1:
        print('Data Set split into train %s and test %s for Cross Validation Purposes'
                            %(ts_train.shape,ts_test.shape))
    ############# Now find the best pdq and PDQ parameters for the model #################
    if not seasonality:
        print('Building a Non Seasonal Model...')
        print('\nFinding best Non Seasonal Parameters:')
        best_p, best_d, best_q, best_bic,seasonality = find_best_pdq_or_PDQ(ts_train, metric,
                                p_max, d_max, q_max, non_seasonal_pdq=None,
                                seasonal_period=None, seasonality=False,verbose=verbose)
        print('\nBest model is: Non Seasonal SARIMAX(%d,%d,%d), %s = %0.3f' %(best_p, best_d,
                                                        best_q,metric, best_bic))
        #### In order to get forecasts to be in the same value ranges of the orig_endogs,
        #### you must  set the simple_differencing = False and the start_params to be the
        #### same as ARIMA.
        #### THat is the only way to ensure that the output of this model is
        #### comparable to other ARIMA models
        bestmodel = sm.tsa.SARIMAX(ts_train, order=(best_p, best_d, best_q),
                                                enforce_stationarity=False,
                                                enforce_invertibility=False,
                                                trend='ct',
                                                start_params=[0, 0, 0, 1],
                                                simple_differencing=False)
    else:
        print(colorful.BOLD +'Building a Seasonal Model...'+colorful.END)
        print(colorful.BOLD +'\n    Finding best Non-Seasonal pdq Parameters:'+colorful.END)
        best_p, best_d, best_q, best_bic,seasonality = find_best_pdq_or_PDQ(ts_train, metric,
                                                p_max, d_max, q_max,
                                                non_seasonal_pdq=None,
                                                seasonal_period=None,
                                                seasonality=False,verbose=verbose)
        print(colorful.BOLD +'\n    Finding best Seasonal PDQ Model Parameters:'+colorful.END)
        best_P, best_D, best_Q, best_bic,seasonality = find_best_pdq_or_PDQ(ts_train, metric,
                                                p_max, d_max, q_max,
                                                non_seasonal_pdq=(best_p,best_d,best_q),
                                                seasonal_period=seasonal_period,
                                                seasonality=True,verbose=verbose)
        if seasonality:
            print('\nBest model is a Seasonal SARIMAX(%d,%d,%d)*(%d,%d,%d,%d), %s = %0.3f' %(
                                                best_p, best_d, best_q, best_P,
                                                best_D, best_Q, seasonal_period,metric, best_bic))
            #### In order to get forecasts to be in the same value ranges of the orig_endogs,
            #### you must set the simple_differencing =False and the start_params to be
            #### the same as ARIMA.
            #### THat is the only way to ensure that the output of this model is
            #### comparable to other ARIMA models
            bestmodel = sm.tsa.statespace.SARIMAX(ts_train, order=(best_p,best_d,best_q),
                                    seasonal_order=(best_P, best_D, best_Q,seasonal_period),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False,
                                    simple_differencing=False, trend='ct',
                                    start_params=[0, 0, 0,1])
        else:
            print('\nBest model is a Non Seasonal SARIMAX(%d,%d,%d)' %(
                                                best_p, best_d, best_q))
            #### In order to get forecasts to be in the same value ranges of the orig_endogs,
            #### you must set the simple_differencing =False and the start_params to be
            #### the same as ARIMA.
            #### THat is the only way to ensure that the output of this model is
            #### comparable to other ARIMA models
            bestmodel = sm.tsa.SARIMAX(ts_train, order=(best_p, best_d, best_q),
                                                    enforce_stationarity=False,
                                                    enforce_invertibility=False,
                                                    trend='ct',
                                                    start_params=[0, 0, 0, 1],
                                                    simple_differencing=False)
    print(colorful.BOLD +'Fitting best SARIMAX model for full data set'+colorful.END)
    try:
        results = bestmodel.fit()
        print('    Best %s metric = %0.1f' %(metric,eval('results.'+metric)))
    except:
        print('Error: Getting Singular Matrix. Please try using other PDQ parameters or turn off Seasonality')
        return bestmodel, None, np.inf, np.inf
    if verbose == 1:
        results.plot_diagnostics(figsize=(16,12))
    ### this is needed for static forecasts ####################
    y_truth = ts_train[:]
    y_forecasted = results.predict(dynamic=False)
    concatenated = pd.concat([y_truth, y_forecasted], axis=1, keys=['original', 'predicted'])
    ### for SARIMAX, you don't have to restore differences since it predicts like actuals.###
    if verbose == 1:
        print('Static Forecasts:')
        print_static_rmse(concatenated['original'].values[best_d:],
                                            concatenated['predicted'].values[best_d:],
                                            verbose=verbose)
    ########### Dynamic One Step Ahead Forecast ###########################
    ### Dynamic Forecats are a better representation of true predictive power
    ## since they only use information from the time series up to a certain point,
    ## and after that, forecasts are generated using values from previous forecasted
    ## time points.
    #################################################################################
    #	Now do dynamic forecast plotting for the last X steps of the data set ######
    if verbose == 1:
        ax = concatenated[['original','predicted']][best_d:].plot(figsize=(16,12))
        startdate = ts_df.index[-forecast_period-1]
        pred_dynamic = results.get_prediction(start=startdate, dynamic=True, full_results=True)
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
        plt.show()
    #	Extract the dynamic predicted and true values of our time series
    y_forecasted = results.forecast(forecast_period)
    if verbose == 1:
        print(results.summary())
    print('Dynamic %d-Period Forecast:' %(forecast_period,))
    rmse, norm_rmse = print_dynamic_rmse(ts_test, y_forecasted, ts_train)
    return results, results.get_forecast(forecast_period,full_results=False).summary_frame(), rmse, norm_rmse
#########################################################################################
from sklearn.metrics import mean_squared_error
def print_static_rmse(actual, predicted, start_from=0,verbose=0):
    """
    this calculates the ratio of the rmse error to the standard deviation of the actuals.
    This ratio should be below 1 for a model to be considered useful.
    The comparison starts from the row indicated in the "start_from" variable.
    """
    rmse = np.sqrt(mean_squared_error(actual[start_from:],predicted[start_from:]))
    std_dev = actual[start_from:].std()
    if verbose == 1:
        print('    RMSE = %0.2f' %rmse)
        print('    Std Deviation of Actuals = %0.2f' %(std_dev))
        print('    Normalized RMSE = %0.1f%%' %(rmse*100/std_dev))
    return rmse, rmse/std_dev
######################################
def predicted_diffs_restored_SARIMAX(original,predicted,periods=1):
    """
    THIS UTILITY IS NEEDED ONLY WHEN WE HAVE SIMPLE DIFFERENCING SET TO TRUE IN SARIMAX!
    The number of periods is equal to the differencing order (d) in the SARIMAX mode.
    SARIMAX predicts a "differenced” prediction only when this simple_differencing=True.
    """
    restored = original.loc[~predicted.isnull()]
    predicted = predicted.loc[~predicted.isnull()]
    restored.iloc[periods:] = predicted[periods:]
    restored = restored.cumsum()
    res = pd.concat([original, predicted, restored], axis=1)
    res.columns = ['original', 'pred_as_diffs', 'predicted']
    res[['original', 'predicted']].plot()
    print_static_rmse(concatenated['original'], concatenated['predicted'])
    return res[['original', 'predicted']]
#######################################
def build_ARIMA_model(ts_df, metric='aic', p_max=12, d_max=1, q_max=12,
                                forecast_period=2, method='mle',verbose=0):
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
                        %(ts_train.shape,ts_test.shape))
    #########################################################################
    if ts_train.dtype=='int64':
        ts_train = ts_train.astype(float)
    for d_val in range(d_min,d_max+1):
        print('\nDifferencing = %d' %d_val)
        results_bic = pd.DataFrame(index=['AR{}'.format(i) for i in range(p_min,p_max+1)],
                                    columns=['MA{}'.format(i) for i in range(q_min,q_max+1)])
        for p_val, q_val in itertools.product(range(p_min,p_max+1), range(q_min,q_max+1)):
            if p_val==0 and d_val==0 and q_val==0:
                results_bic.loc['AR{}'.format(p_val), 'MA{}'.format(q_val)] = np.nan
                continue
            else:
                try:
                    model = sm.tsa.ARIMA(ts_train,order=(p_val, d_val, q_val))
                    results = model.fit(transparams=False,method=method)
                    results_bic.loc['AR{}'.format(p_val), 'MA{}'.format(q_val)] = eval('results.'+metric)
                    if iteration % 10 == 0:
                        print(' Iteration %d completed...' %iteration)
                    iteration += 1
                    if iteration >= 100:
                        print('    Ending Iterations at %d' %iteration)
                        break
                except:
                    iteration += 1
                    continue
        results_bic = results_bic[results_bic.columns].astype(float)
        interim_d = copy.deepcopy(d_val)
        interim_p, interim_q, interim_bic = find_lowest_pq(results_bic)
        if verbose == 1:
            fig, ax = plt.subplots(figsize=(20,10))
            ax = sns.heatmap(results_bic,
                            mask=results_bic.isnull(),
                            ax=ax,
                            annot=True,
                            fmt='.0f')
            ax.set_title(metric)
        results_dict[str(interim_p)+' '+str(interim_d)+' '+str(interim_q)] = interim_bic
    best_bic = min(results_dict.items(), key=operator.itemgetter(1))[1]
    best_pdq = min(results_dict.items(), key=operator.itemgetter(1))[0]
    best_p = int(best_pdq.split(' ')[0])
    best_d = int(best_pdq.split(' ')[1])
    best_q = int(best_pdq.split(' ')[2])
    print('\nBest model is: Non Seasonal ARIMA(%d,%d,%d), %s = %0.3f' %(best_p, best_d, best_q,metric, best_bic))
    bestmodel = sm.tsa.ARIMA(ts_train,order=(best_p, best_d, best_q))
    print('####    Fitting best model for full data set now. Will take time... ######')
    try:
        results = bestmodel.fit(transparams=True, method=method)
    except:
        results = bestmodel.fit(transparams=False, method=method)
    ### this is needed for static forecasts ####################
    y_truth = ts_train[:]
    y_forecasted = results.predict()
    concatenated = pd.concat([y_truth, y_forecasted], axis=1, keys=['original', 'predicted'])
    if best_d ==0:
        #### Do this for ARIMA only ######
        ###  If there is no differencing DO NOT use predict_type since it will give an error = do not use "linear".
        print('Static Forecasts:' )
        print_static_rmse(concatenated['original'].values, concatenated ['predicted'].values, best_d)
        startdate = ts_df.index[-forecast_period]
        enddate = ts_df.index[-1]
        pred_dynamic = results.predict( start=startdate, end=enddate, dynamic=True)
        if verbose == 1:
            ax = concatenated[['original','predicted']][best_d:].plot()
            pred_dynamic.plot(label='Dynamic Forecast', ax=ax, figsize=(15,5))
            print('Dynamic %d-period Forecasts:' %(forecast_period,))
            plt.legend()
            plt.show()
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
        startdate = ts_df.index[-forecast_period]
        enddate = ts_df.index[-1]
        pred_dynamic = results.predict(typ=pred_type, start=startdate, end=enddate, dynamic=True)
        pred_dynamic[pd.to_datetime((pred_dynamic.index-best_d).values[0])] = y_truth[
                                            pd.to_datetime((pred_dynamic.index-best_d).values[0])]
        pred_dynamic.sort_index(inplace=True)
        print('\nDynamic %d-period Forecasts:' %forecast_period)
        if verbose == 1:
            ax = concatenated.plot()
            pred_dynamic.plot(label='Dynamic Forecast', ax=ax, figsize=(15,5))
            ax.set_xlabel('Date')
            ax.set_ylabel('Values')
            plt.legend()
            plt.show()
    #### Don't know if we need to fit again! ############
    results = bestmodel.fit()
    if verbose == 1:
        results.plot_diagnostics(figsize=(16,12))
    res_frame = pd.DataFrame([results.forecast(forecast_period)[0],results.forecast(forecast_period)[1],
                                                results.forecast(forecast_period)[2]],
                                                index=['mean','mean_se','mean_ci'],
                                                columns=['Forecast_'+str(x) for x
                                                in range(1,forecast_period+1)]).T
    res_frame['mean_ci_lower'] = res_frame['mean_ci'].map(lambda x: x[0])
    res_frame['mean_ci_upper'] = res_frame['mean_ci'].map(lambda x: x[1])
    res_frame.drop('mean_ci',axis=1,inplace=True)
    if verbose == 1:
        print('Model Forecast(s):\n', res_frame)
    rmse, norm_rmse = print_dynamic_rmse(ts_test, pred_dynamic, ts_train)
    return results, res_frame, rmse, norm_rmse
####################
def predicted_diffs_restored_ARIMA(actuals,predicted,periods=1):
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
        restored.iloc[periods:] = predicted [periods:]
        restored = restored[(periods-1):].cumsum()
    res = pd.concat([actuals, predicted, restored], axis=1)
    res.columns = ['original', 'pred_as_diffs', 'predicted']
    print_static_rmse(res['original'].values, res['predicted'].values,periods-1)
    return res[['original', 'predicted']]
############################################
def find_max_min_value_in_a_dataframe(df, max_min='min'):
    """
    This returns the lowest or highest value in a df and its row value where it can be found.
    Unfortunately, it does not return the column where it is found. So not used much.
    """
    if max_min == 'min':
        return df.loc[:, list(df)].min(axis=1).min(), df.loc[:, list(df)].min(axis=1).idxmin()
    else:
        return df.loc[:, list(df)].max(axis=1).max(), df.loc[:, list(df)].min(axis=1).idxmax()
##########################################
def find_lowest_pq(df):
    """
    This finds the row and column numbers of the lowest or highest value in a dataframe. All it needs is numeric values.
    It will return the row and column together as a string, you will have to split it into two.
    It will also return the lowest value in the dataframe by default but you can change it to "max".
    """
    dicti = {}
    for ma in list(df):
        try:
            dicti [ma +' '+ df[ma].idxmin()] = df[ma].sort_values()[0]
        except:
            pass
    lowest_bic = min(dicti.items(), key=operator.itemgetter(1))[1]
    lowest_pq = min(dicti.items(), key=operator.itemgetter(1))[0]
    ma_q = int(lowest_pq.split(' ')[0][2:])
    ar_p = int(lowest_pq.split(' ')[1][2:])
    print('    Best AR order p = %d, MA order q = %d, Interim metric = %0.3f' %(ar_p, ma_q, lowest_bic))
    return ar_p, ma_q, lowest_bic
##############################
def build_VAR_model(df, criteria, forecast_period=2, p_max=3,q_max=3,verbose=0):
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
                            %(ts_train.shape,ts_test.shape))
    # It is assumed that the first column of the dataframe is the target variable ####
    ### make sure that is the case before doing this program ####################
    i = 1
    results_dict = {}
    for d_val in range(1,dmax):
        y_train = ts_train.iloc[:,[0,d_val]]
        print('\nAdditional Variable in VAR model = %s' %cols[d_val])
        info_criteria = pd.DataFrame(index=['AR{}'.format(i) for i in range(0,p_max+1)],
                            columns=['MA{}'.format(i) for i in range(0,q_max+1)])
        for p_val, q_val in itertools.product(range(0,p_max+1),range(0,q_max+1)):
            if p_val==0 and q_val==0:
                info_criteria.loc['AR{}'.format(p_val), 'MA{}'.format(q_val)] = np.nan
                print(' Iteration %d completed' %i)
                i += 1
            else:
                try:
                    model = sm.tsa.VARMAX(y_train, order=(p_val,q_val), trend='c')
                    model = model.fit(max_iter=1000, displ=False)
                    info_criteria.loc['AR{}'.format(p_val), 'MA{}'.format(q_val)] = eval('model.'+criteria)
                    print(' Iteration %d completed' %i)
                    i+=1
                except:
                    i += 1
                    print(' Iteration %d completed' %i)
        info_criteria = info_criteria[info_criteria.columns].astype(float)
        interim_d = copy.deepcopy(d_val)
        interim_p, interim_q, interim_bic = find_lowest_pq(info_criteria)
        if verbose == 1:
            fig, ax = plt.subplots(figsize=(20,10))
            ax = sns.heatmap(info_criteria,
                    mask=info_criteria.isnull(),
                    ax=ax,
                    annot=True,
                    fmt='.0f',
                    )
            ax.set_title(criteria)
        results_dict[str(interim_p)+' '+str(interim_d)+' '+str(interim_q)] = interim_bic
    best_bic = min(results_dict.items(), key=operator.itemgetter(1))[1]
    best_pdq = min(results_dict.items(), key=operator.itemgetter(1))[0]
    best_p = int(best_pdq.split(' ')[0])
    best_d = int(best_pdq.split(' ')[1])
    best_q = int(best_pdq.split(' ')[2])
    print('Best variable selected for VAR: %s' %ts_train.columns.tolist()[best_d])
    y_train = ts_train.iloc[:,[0, best_d]]
    bestmodel = sm.tsa.VARMAX(y_train, order=(best_p,best_q), trend='c')
    bestmodel = bestmodel.fit()
    if verbose == 1:
        bestmodel.plot_diagnostics(figsize=(16,12))
        ax = bestmodel.impulse_responses(12, orthogonalized=True).plot(figsize=(12,4))
        ax.set(xlabel='Time Steps', title='Impulse Response Functions')
    res2 = bestmodel.get_forecast(forecast_period)
    res2_df = res2.summary_frame()
    rmse, norm_rmse =  print_dynamic_rmse(ts_test.iloc[:,0], res2_df['mean'].values, ts_train.iloc[:,0])
    return bestmodel, res2_df, rmse, norm_rmse
#########################################################
def build_pyflux_model(df,target,ar=12,ma=12,integ=1,forecast_period=2,fitmethod='MLE',
                                            nsims=100, score_type='rmse', verbose=0):
    """
    Build a quick pyflux model with default parameters for AR, MA and I terms in ARIMA.
    You can build a rolling forecast using the rolling_forecast parameter.
    PyFlux is a fiendishly complicated program with very poor documentation.
    I had to dig deep into the API to figure these things out especially the
    """
    import pyflux as pf
    ts_df = df[:]
    ##############################################################################
    ts_train = ts_df[:-forecast_period]
    ts_test = ts_df[-forecast_period:]
    if verbose == 1:
        print('Data Set split into train %s and test %s for Cross Validation Purposes'
                            %(ts_train.shape,ts_test.shape))
    #####################################################################################################
    if integ > 1:
        print('    Setting "integration"=1 since differenced predictions > 1 are difficult to interpret')
        integ = 1
    if fitmethod == 'M-H':
        print('    Assuming number of simulations = %d' %nsims)
    ####################################################################################################
    ###### define p,d,q parameters here ####################
    p = range(0,ar+1)
    q = range(0,ma+1)
    d = range(0,integ+1)  ### dont do much more than 1 differencing in PyFlux models since its hard to undo
    #### Generate all different combinations of p,d,q triplets ######
    pdq = list(itertools.product(p,d,q))
    eval_metrics = {}
    print('Cycling through various (p,d,q) parameters')
    for param in pdq:
        if verbose == 1:
            print('.', end="")
        model = pf.ARIMA(data=ts_train, ar=param[0], integ=param[1], ma=param[2], target=target)
        try:
            if fitmethod == 'MLE':
                x = model.fit()
            elif fitmethod == 'M-H':
                x = model.fit('M-H', nsims=nsims)
        except:
            x = model.fit('MLE')
        mu, actuals = model._model(model.latent_variables.get_z_values())
        predicted = model.link(mu)
        rmse, norm_rmse = print_static_rmse(actuals,predicted)
        if score_type == 'rmse':
            eval_metrics[param] = rmse
        else:
            eval_metrics[param] = norm_rmse
    bestpdq = min(eval_metrics.items(), key=operator.itemgetter(1))[0]
    print('\nBest Params Selected (based on %s): %s' %(score_type, bestpdq))
    bestmodel = pf.ARIMA(data=ts_train, ar= bestpdq[0], integ=bestpdq[1], ma=bestpdq[2], target=target)
    x = bestmodel.fit()
    if verbose == 1:
        bestmodel.plot_fit(figsize=(15,5))
    #model.plot_predict_is(h=forecast_period,fit_once=False,fit_method=fitmethod)
    if verbose == 1:
        x.summary()
        n = int(0.5*len(df))
        bestmodel.plot_predict(h=forecast_period,past_values=n,intervals=True, figsize=(15,5))
    forecast_df = bestmodel.predict(forecast_period, intervals=True)
    mu, actuals = bestmodel._model(bestmodel.latent_variables.get_z_values())
    predicted = bestmodel.link(mu)
    print('Dynamic %d-period Forecasts:' %forecast_period)
    pdb.set_trace()
    if bestpdq[1] == 1:
        mod_target = 'Differenced '+target
        res = restore_differenced_predictions(ts_test[target].values,forecast_df[mod_target],
                                    ts_train[target][-1:])
        rmse, norm_rmse = print_dynamic_rmse(ts_test[target].values, res, ts_train[target])
    else:
        rmse, norm_rmse = print_dynamic_rmse(ts_test[target].values,forecast_df[target].values, ts_train[target])
    return bestmodel, forecast_df, rmse, norm_rmse
######################################################
def restore_differenced_predictions(actuals, predicted,start_value,func=None,periods=1,diff_yes=True):
    try:
        restored = pd.Series(index=start_value.index)
        restored.ix[start_value.ix[:periods].index] =  start_value.values[:periods]
        rest = restored.ix[predicted.index]
        restored = pd.Series(np.r_[restored,rest],index=np.r_[start_value.index,rest.index])
        restored.ix[predicted.index] = predicted.values
        restored = restored[(periods-1):].cumsum()
        if func:
            restored = eval('np.'+func+'(restored)')
        return restored[periods:]
    except:
        restored = start_value.values+predicted
        if func:
            restored = eval('np.'+func+'(restored)')
        return restored
#########################################################
from sklearn.model_selection import TimeSeriesSplit
def cross_validation_time_series(model, df, preds, target,n_times=10,verbose=0):
    """
    This splits a time series data frame "n" times as specified in the input (default=10)
    Initially it will start with a certain number of rows in train but it will gradually
    increase train size in steps (which it will calculate automatically) while the
    number of test rows will remain the same (though their content will vary).
    This utility is based on sklearn's time_series_split()
    """
    if n_times > 10:
        print('More than 10 splits is not recommended. Setting n_times to 10')
        n_times = 10
    splits = TimeSeriesSplit(n_splits=n_times)
    index = 0
    X = df[preds].values
    y = df[target].values
    non_df = {}
    rmse_list = []
    for train_index, test_index in splits.split(X):
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]
        if verbose == 1:
            print('Iteration %d: Total Observations = %d' %(index,len(X_train)+len(X_test)))
            print('    Training Index %d Observations: %s' %(len(train_index),train_index))
            print('    Testing Index %d Observations: %s' %(len(test_index),test_index))
        model.fit(X_train, y_train)
        rmse = print_rmse(y_test, model.predict(X_test))
        rmse_list.append(rmse)
        norm_rmse = rmse/y_test.std()
        print('     Split %d: Normalized RMSE = %0.2f' %(norm_rmse))
        non_df[index] = norm_rmse
        index += 1
    non_df = pd.Series(non_df)
    non_df.plot()
    ave_norm_rmse = np.mean(rmse_list)/y.std()
    print('Normalized RMSE over  entire data after %d splits = 0.2f' %(index,ave_norm_rmse))
    return ave_norm_rmse
##########################################################
def rolling_validation_time_series(model, df, preds, target,train_size=0,
                                                    test_size=0, verbose=0):
    """
    This utility uses a Walk Forward or Rolling Period time series cross validation method.
    Initially it will start with a minimum number of observations to train the model.
    It then gradually increases the train size in steps (which it will calculate automatically)
    while fixing the number of test rows the same (though their content will vary).
    Once the train+test series exceeds the number of rows in data set, it stops.
    It does  not use SKLearn's Time Series Split. You need to provide the initial sizes
    of train and test and it will take care of the rest.
    """
    df = df[:]
    index = 0
    X = df[preds].values
    y = df[target].values
    non_df = {}
    rmse_list = []
    if train_size == 0:
        train_size = np.int(np.ceil(len(y)/2))
    if test_size == 0:
        test_size = np.int(np.ceil(len(y)/4))
    step_size = np.int(np.ceil(test_size/10))
    n_records = len(X)
    ### This contains the start point of test size for each K-Fold in time series
    test_list = np.floor(np.linspace(train_size,n_records-1,5)).tolist()
    for i in range(4):
        train_size = np.int(test_list[i])
        test_size = np.int(test_list[i+1] - test_list[i])
        X_train, X_test = X[:train_size],X[train_size:train_size+test_size]
        y_train, y_test = y[:train_size],y[train_size:train_size+test_size]
        model.fit(X_train, y_train)
        if i == 0:
            ### Since both start and end points are included, you have to subtract 1 from index in this
            df.loc[:train_size-1,'predictions'] = y[:train_size]
            df.loc[train_size:train_size+test_size-1,'predictions'] = model.predict(X_test)
        elif i == 3:
            test_size = np.int(len(X) - train_size)
            X_train, X_test = X[:train_size],X[train_size:train_size+test_size]
            y_train, y_test = y[:train_size],y[train_size:train_size+test_size]
            df.loc[train_size:train_size+test_size,'predictions'] = model.predict(X_test)
        else:
            df.loc[train_size:train_size+test_size-1,'predictions'] = model.predict(X_test)
        if len(y_train) + len(y_test) >= df.shape[0]:
            if verbose:
                print('Iteration %d: Observations:%d' %(index+1,len(X_train)+len(X_test)))
                print('    Train Size=%d, Test Size=%d' %(len(y_train),len(y_test)))
            rmse = print_rmse(y_test, model.predict(X_test))
            norm_rmse = rmse/y_test.std()
            non_df[i] = rmse
            if verbose:
                print('Normalized RMSE = %0.2f' %norm_rmse)
            non_df = pd.Series(non_df)
            weighted_ave_rmse = np.average(non_df.values,weights=non_df.index,axis=0)
            print('\nWeighted Average of RMSE (%d iterations) = %0.2f\n    Normalized Wtd Aver. RMSE (using std dev) = %0.2f'
                                %(index+1, weighted_ave_rmse,weighted_ave_rmse/y[:].std()))
            #############################
            if verbose == 1 or verbose == 2:
                fig, ax1 = plt.subplots(nrows=1,ncols=1,figsize=(12,8))
                ax1.plot(df[target],label='In-Sample Data', linestyle='-')
                ax1.plot(df['predictions'],'g',alpha=0.6,label='Rolling Forecast')
                ax1.set_xlabel('Time')
                ax1.set_ylabel('Values')
                ax1.legend(loc='best')
            return weighted_ave_rmse, weighted_ave_rmse/y[:].std(), df
        else:
            if verbose:
                print('Iteration %d: Observations:%d' %(index+1,len(X_train)+len(X_test)))
                print('    Train Size=%d, Test Size=%d' %(len(y_train),len(y_test)))
            rmse = print_rmse(y_test, model.predict(X_test))
            norm_rmse = rmse/y_test.std()
            non_df[i] = rmse
            if verbose:
                print('Normalized RMSE = %0.2f' %norm_rmse)
            index += 1

##########################################################
from sklearn.metrics import mean_squared_error,mean_absolute_error
def print_rmse(y, y_hat):
    """
    Calculating Root Mean Square Error https://en.wikipedia.org/wiki/Root-mean-square_deviation
    """
    mse = np.mean((y - y_hat)**2)
    return np.sqrt(mse)

def print_mape(y, y_hat):
    """
    Calculating Mean Absolute Percent Error https://en.wikipedia.org/wiki/Mean_absolute_percentage_error
    """
    perc_err = (100*(y - y_hat))/y
    return np.mean(abs(perc_err))

def print_normalized_rmse(actuals, predicted,start_from=0):
    """
    This utility calculates rmse between actuals and predicted. However, it does one more.
    If the original is given, it calculates Normalized RMSE using the original array's std deviation.
    """
    actuals = actuals[start_from:]
    predicted = predicted[start_from:]
    rmse = np.sqrt(np.mean(mean_squared_error(actuals,predicted)))
    norm_rmse = rmse/actuals.std()
    print('RMSE = {:,.2f}'.format(rmse))
    print('Std Deviation of Actuals = {:,.2f}'.format(actuals.std()))
    print('Normalized RMSE = %0.0f%%' %(100*norm_rmse))
    return rmse, norm_rmse

def print_dynamic_rmse(actuals, predicted, original):
    """
    This utility calculates rmse between actuals and predicted. However, it does one more.
    Since in dynamic forecast, we need the longer original, it calculates Normalized RMSE
    using the original array's std deviation. That way, the forecast of 2 values does not
    result in a larger Normalized RMSE since the std deviation of 2 values will be v small.
    """
    rmse = np.sqrt(np.mean((actuals - predicted)**2))
    norm_rmse = rmse/original.std()
    print('    RMSE = {:,.2f}'.format(rmse))
    print('    Std Deviation of Originals = {:,.2f}'.format(original.std()))
    print('    Normalized RMSE = %0.0f%%' %(100*norm_rmse))
    return rmse, norm_rmse

def print_ts_model_stats(actuals, predicted, number_as_percentage=100):
    """
    This program prints and returns MAE, RMSE, MAPE.
    If you like the MAE and RMSE as a percentage of something, just give that number
    in the input as "number_as_percentage" and it will return the MAE and RMSE as a
    ratio of that number. Returns MAE, MAE_as_percentage, and RMSE_as_percentage
    """
    #print(len(actuals))
    #print(len(predicted))
    plt.figure(figsize=(15,8))
    dfplot = pd.DataFrame([predicted,actuals]).T
    dfplot.columns = ['Forecast','Actual']
    plt.plot(dfplot)
    plt.legend(['Forecast','Actual'])
    mae = mean_absolute_error(actuals, predicted)
    mae_asp = (mean_absolute_error(actuals, predicted)/number_as_percentage)*100
    rmse_asp = (np.sqrt(mean_squared_error(actuals,predicted))/number_as_percentage)*100
    print('MAE (%% AUM) = %0.2f%%' %mae_asp)
    print('RMSE (%% AUM) = %0.2f%%' %rmse_asp)
    print('MAE (as %% Actual) = %0.2f%%' %(mae/abs(actuals).mean()*100))
    _ = print_mape(actuals, predicted)
    rmse = print_rmse(actuals, predicted)
    mape = print_mape(actuals, predicted)
    print("MAPE = %0.0f%%" %(mape))
    # Normalized RMSE print('RMSE = {:,.Of}'.format(rmse))
    print('Normalized RMSE (MinMax) = %0.0f%%' %(100*rmse/abs(actuals.max()-actuals.min())))
    print('Normalized RMSE = %0.0f%%' %(100*rmse/actuals.std()))
    return mae, mae_asp, rmse_asp
###################################################
# Re-run the above statistical tests, and more. To be used when selecting viable models.
def ts_model_validation(model_results):
    """
    Once you have built a time series model, how to validate it. This utility attempts to.
    This is only done on SARIMAX models from statsmodels. Don't try it on other models.
    The input is model_results which is the variable assigned to the model.fit() method.
    """
    het_method='breakvar'
    norm_method='jarquebera'
    sercor_method='ljungbox'
    ########################
    (het_stat, het_p) = model_results.test_heteroskedasticity(het_method)[0]
    norm_stat, norm_p, skew, kurtosis = model_results.test_normality(norm_method)[0]
    sercor_stat, sercor_p = model_results.test_serial_correlation(method=sercor_method)[0]
    sercor_stat = sercor_stat[-1] # last number for the largest lag
    sercor_p = sercor_p[-1] # last number for the largest lag

    # Run Durbin-Watson test on the standardized residuals.
    # The statistic is approximately equal to 2*(1-r), where r is the sample autocorrelation of the residuals.
    # Thus, for r == 0, indicating no serial correlation, the test statistic equals 2.
    # This statistic will always be between 0 and 4. The closer to 0 the statistic,
    # the more evidence for positive serial correlation. The closer to 4,
    # the more evidence for negative serial correlation.
    # Essentially, below 1 or above 3 is bad.
    dw = sm.stats.stattools.durbin_watson(model_results.filter_results.standardized_forecasts_error[0, model_results.loglikelihood_burn:])

    # check whether roots are outside the unit circle (we want them to be);
    # will be True when AR is not used (i.e., AR order = 0)
    arroots_outside_unit_circle = np.all(np.abs(model_results.arroots) > 1)
    # will be True when MA is not used (i.e., MA order = 0)
    maroots_outside_unit_circle = np.all(np.abs(model_results.maroots) > 1)

    print('Test heteroskedasticity of residuals ({}): stat={:.3f}, p={:.3f}'.format(het_method, het_stat, het_p));
    print('\nTest normality of residuals ({}): stat={:.3f}, p={:.3f}'.format(norm_method, norm_stat, norm_p));
    print('\nTest serial correlation of residuals ({}): stat={:.3f}, p={:.3f}'.format(sercor_method, sercor_stat, sercor_p));
    print('\nDurbin-Watson test on residuals: d={:.2f}\n\t(NB: 2 means no serial correlation, 0=pos, 4=neg)'.format(dw))
    print('\nTest for all AR roots outside unit circle (>1): {}'.format(arroots_outside_unit_circle))
    print('\nTest for all MA roots outside unit circle (>1): {}'.format(maroots_outside_unit_circle))
###############################
def time_series_split(ts_df):
    """
    This utility splits any dataframe sent as a time series split using the sklearn function.
    """
    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=2)
    train_index, test_index = list(tscv.split(ts_df))[1][0],list(tscv.split(ts_df))[1][1]
    ts_train, ts_test = ts_df[ts_df.index.isin(train_index)], ts_df[
                        ts_df.index.isin(test_index)]
    print(ts_train.shape, ts_test.shape)
    return ts_train, ts_test
############################################
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import GridSearchCV

###########################################################
def build_prophet_model(ts_df, time_col, target, forecast_period,
                            score_type,verbose,conf_int):
    """
    Build a Time Series Model using Facebook Prophet which is a powerful model.
    """
    from fbprophet import Prophet
    df = ts_df[:]
    try:
        df[time_col].head()
    except:
        #### THis happens when time_col is not found but it's actually the index. In that case, reset index
        df = ts_df.reset_index()
    df.rename(columns={time_col:'ds', target:'y'},inplace=True)    
    actual = 'y'
    timecol = 'ds'
    dft = df[[timecol,actual]]
    ##### For most Financial time series data, 80% conf interval is enough...
    print('    Fit-Predict data (shape=%s) with Confidence Interval = %0.2f...' %(dft.shape,conf_int))
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
    print('Building Forecast dataframe. Forecast Period = %d' %forecast_period)
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
    model.plot_components(forecast);
    rmse, norm_rmse = print_dynamic_rmse(dfa['y'],dfa['yhat'],dfa['y'])
    #submit = dfplot[-forecast_period:]
    #submit.drop('Actuals',axis=1,inplace=True)
    #submit.rename(columns={'yhat':target},inplace=True)
    #print('Forecast Data frame size %s ready to submit' %(submit.shape,))
    return model, forecast, rmse, norm_rmse
###########################################################################################
def plot_prophet(dft,forecastdf):
    """
    This is a different way of plotting Prophet charts as described in the following article:
    Source: https://nextjournal.com/viebel/forecasting-time-series-data-with-prophet
    Reproduced with gratitude to the author.
    """
    dft = copy.deepcopy(dft)
    forecastdf = copy.deepcopy(forecastdf)
    dft.set_index('ds',inplace=True)
    forecastdf.set_index('ds',inplace=True)
    dft.index = pd.to_datetime(dft.index)
    connect_date = dft.index[-2]
    mask = (forecastdf.index > connect_date)
    predict_df = forecastdf.loc[mask]
    viz_df = dft.join(predict_df[['yhat','yhat_lower','yhat_upper']],
                     how='outer')
    fig,ax1 = plt.subplots(figsize=(20,10))
    ax1.plot(viz_df['y'],color='red')
    ax1.plot(viz_df['yhat'],color='green')
    ax1.fill_between(viz_df.index, viz_df['yhat_lower'],viz_df['yhat_upper'],
                    alpha=0.2,color="darkgreen")
    ax1.set_title('Actuals (Red) vs Forecast (Green)')
    ax1.set_ylabel('Values')
    ax1.set_xlabel('Date Time')
    plt.show();
    return viz_df
###########################################################
def Auto_Timeseries(trainfile, ts_column, sep=',', target=None, score_type='rmse', forecast_period=2,
                        timeinterval='', non_seasonal_pdq=None, seasonality=False,
                        seasonal_period=12, seasonal_PDQ=None, conf_int=0.95, model_type = "stats",
                        verbose=0):
    """
    ##################################################################################################
    AUTO_TIMESERIES IS A COMPLEX MODEL BUILDING UTILITY FOR TIME SERIES DATA. SINCE IT AUTOMATES MANY
    TASKS INVOLVED IN A COMPLEX ENDEAVOR, IT ASSUMES MANY INTELLIGENT DEFAULTS. BUT YOU CAN CHANGE THEM.
    Auto_Timeseries will rapidly build predictive models based on Statsmodels ARIMA, Seasonal ARIMA
    and Scikit-Learn ML. It will automatically select the BEST model which gives best score specified.
    It will return the best model and a dataframe containing predictions for forecast_period (default=2).
    #####################################################################################################
    INPUT:
    #####################################################################################################
    trainfile: name of the file along with its data path or a dataframe. It accepts both.
    ts_column: name of the datetime column in your dataset (it could be name or number)
    target: name of the column you are trying to predict. Target could also be the only column in your data
    score_type: 'rmse' is the default. You can choose among "mae", "mse" and "rmse". 
    forecast_period: default is 2. How many periods out do you want to forecast? It should be an integer
    timeinterval: default is "Month". What is the time period in your data set. Options are: "days",
    model_type: default is "stats". Choice is between "stats", "prophet" and "ml". "All" will build all.
        - stats will build statsmodels based ARIMA< SARIMAX and VAR models
        - ml will build a machine learning model using Random Forests provided explanatory vars are given
        - prophet will build a model using FB Prophet -> this means you must have FB Prophet installed
        - all will build all the above models which may take a long time for large data sets. 
    We recommend that you choose a small sample from your data set bedfore attempting to run entire data.
    #####################################################################################################
    and the evaluation metric so it can select the best model. Currently only 2 are supported: RMSE and
    Normalized RMSE (ratio of RMSE to the standard deviation of actuals). Other eval metrics will be soon.
    the target variable you are trying to predict (if there is more than one variable in your data set),
    and the time interval that is in the data. If your data is in a different time interval than given,
    Auto_Timeseries will automatically resample your data to the given time interval and learn to make
    predictions. Notice that except for filename and ts_column which are required, all others are optional.
    Note that optionally you can give a separator for the data in your file. Default is comman (",").
    "timeinterval" options are: 'Days', 'Weeks', 'Months', 'Qtr', 'Year', 'Minutes', 'Hours', 'Seconds'.
    Optionally, you can give seasonal_period as any integer that measures the seasonality in the data.
    If not, seasonal_period is assumed automatically as follows: Months = 12, Days = 30, Weeks = 52,
    Qtr = 4, Year = 1, Hours = 24, Minutes = 60 and Seconds = 60.
    If you want to give your own order, please input it as non_seasonal_pdq and seasonal_PDQ in the input
    as tuples. For example, seasonal_PDQ = (2,1,2) and non_seasonal_pdq = (0,0,3). It will accept only tuples.
    The defaul is None and Auto_Timeseries will automatically search for the best p,d,q (for Non Seasonal)
    and P, D, Q (for Seasonal) orders by searching for all parameters from 0 to 12 for each value of
    p,d,q and 0-3 for each P, Q and 0-1 for D.
    #####################################################################################################
    """
    start_time = time.time()
    ##### Best hyper-parameters in statsmodels chosen using the best aic, bic or whatever. Select here.
    stats_scoring = 'aic'
    seed = 99
    ### If run_prophet is set to True, then only 1 model will be run and that is FB Prophet ##
    lag = copy.deepcopy(forecast_period)-1
    if type(non_seasonal_pdq) == tuple:
        p_max = non_seasonal_pdq[0]
        d_max = non_seasonal_pdq[1]
        q_max = non_seasonal_pdq[2]
    else:
        p_max = 12
        d_max = 1
        q_max = 12
    ################################
    if type(seasonal_PDQ) == tuple:
        seasonal_order = copy.deepcopy(seasonal_PDQ)
    else:
        seasonal_order = (3,1,3)
    ########## This is where we start the loading of the data file ######################
    if isinstance(trainfile, str):
        if trainfile != '':
            try:
                ts_df = load_ts_data(trainfile, ts_column, sep, target)
                print('    File loaded successfully. Shape of data set = %s' %(ts_df.shape,))
            except:
                print('File could not be loaded. Check the path or filename and try again')
                return
        elif isinstance(trainfile, pd.DataFrame):
            print('Input is data frame. Performing Time Series Analysis')
            ts_df = load_ts_data(trainfile, ts_column, sep, target)
        else:
            print('File name is an empty string. Please check your input and try again')
            return
    else:
        print('Dataframe given as input. Analyzing Time Series data...')
        ts_df = copy.deepcopy(trainfile)
    df_orig = copy.deepcopy(ts_df)
    if ts_df.shape[1] == 1:
        ### If there is only one column, you assume that to be the target column ####
        target = list(ts_df)[0]
    if not isinstance(ts_column, str):
        ### If ts_column is a number, then it means you need to convert it to a named variable
        ts_column = list(ts_df)[ts_column]
    if isinstance(target,list):
        target = target[0]
        print('    Taking the first column in target list as Target variable = %s' %target)
    else:
        print('    Target variable = %s' %target)
    preds = [x for x in list(ts_df) if x not in [ts_column,target]]
    ##################################################################################################
    ### Turn the time series index into a variable and calculate the difference. 
    ### If the difference is not in days, then it is a hourly or minute based time series
    ### If the difference a multiple of days, then test it for weekly, monthly, qtrly, annual etc.
    ##################################################################################################
    if ts_df.index.dtype=='int' or ts_df.index.dtype=='float':
        ### You must convert the ts_df index into a date-time series using the ts_column given ####
        ts_df = ts_df.set_index(ts_column)
    ts_index = ts_df.index
    ################    IF TIME INTERVAL IS NOT GIVEN DO THIS   ########################
    #######   This is where the program tries to tease out the time period in the data set ###########
    ##################################################################################################
    if timeinterval == '':
        ts_index = pd.to_datetime(ts_df.index)
        diff = (ts_index[1] - ts_index[0]).to_pytimedelta()
        diffdays = diff.days
        diffsecs = diff.seconds
        if diffsecs == 0:
            diff_in_hours = 0
            diff_in_days = abs(diffdays)
        else:
            diff_in_hours = abs(diffdays*24*3600+diffsecs)/3600    
        if diff_in_hours == 0 and diff_in_days >= 1:
            print('Time series input in days = %s' %diff_in_days)
            if diff_in_days == 7:
                print('it is a Weekly time series.')
                timeinterval = 'weeks'
            elif diff_in_days == 1:
                print('it is a Daily time series.')
                timeinterval = 'days'
            elif diff_in_days >= 28 and diff_in_days < 89:
                print('it is a Monthly time series.')
                timeinterval = 'months'
            elif diff_in_days >= 89 and diff_in_days < 178:
                print('it is a Quarterly time series.')
                timeinterval = 'qtr'
            elif diff_in_days >= 178 and diff_in_days < 360:
                print('it is a Semi Annual time series.')
                timeinterval = 'qtr'
            elif diff_in_days >= 360:
                print('it is an Annual time series.')
                timeinterval = 'years'
            else:
                print('Time Series time delta is unknown')
                return
        if diff_in_days == 0:
            if diff_in_hours == 0:
                print('Time series input in Minutes or Seconds = %s' %diff_in_hours)
                print('it is a Minute time series.')
                timeinterval = 'minutes'
            elif diff_in_hours >= 1:
                print('it is an Hourly time series.')
                timeinterval = 'hours'
            else:
                print('It is an Unknown Time Series delta')
                return
    else:
        print('Time Interval is given as %s' %timeinterval)
    ################# This is where you test the data and find the time interval #######
    if timeinterval == 'Months' or timeinterval == 'month' or timeinterval == 'months':
        timeinterval = 'months'
        seasonal_period = 12
    elif timeinterval == 'Days' or timeinterval == 'daily' or timeinterval == 'days':
        timeinterval = 'days'
        seasonal_period = 30
        ts_df = ts_df.resample('D').sum()
    elif timeinterval == 'Weeks' or timeinterval == 'weekly' or timeinterval == 'weeks':
        timeinterval = 'weeks'
        seasonal_period = 52
    elif timeinterval == 'Qtr' or timeinterval == 'quarter' or timeinterval == 'qtr':
        timeinterval = 'qtr'
        seasonal_period = 4
    elif timeinterval == 'Year' or timeinterval == 'year' or timeinterval == 'years' or timeinterval == 'annual':
        timeinterval = 'years'
        seasonal_period = 1
    elif timeinterval == 'Hours' or timeinterval == 'hours' or timeinterval == 'hourly':
        timeinterval = 'hours'
        seasonal_period = 24
    elif timeinterval == 'Minutes' or timeinterval == 'minute' or timeinterval == 'minutes':
        timeinterval = 'minutes'
        seasonal_period = 60
    elif timeinterval == 'Seconds' or timeinterval == 'second' or timeinterval == 'seconds':
        timeinterval = 'seconds'
        seasonal_period = 60
    else:
        timeinterval = 'months'
        seasonal_period = 12
    ########################### This is where we store all models in a nested dictionary ##########
    mldict = lambda: defaultdict(mldict)
    ml_dict = mldict()
    ######### This is when you need to use FB Prophet ###################################
    ### When the time interval given does not match the tested_timeinterval, then use FB.
    #### Also when the number of rows in data set is very large, use FB Prophet, It is fast.
    #########                 FB Prophet              ###################################
    if model_type.lower() == 'prophet' or model_type.lower() == 'all':
        name = 'FB_Prophet'
        print(colorful.BOLD + '\nRunning Facebook Prophet Model...' + colorful.END)
        try:
            #### If FB prophet needs to run, it needs to be installed. Check it here ###
            model, forecast_df, rmse, norm_rmse = build_prophet_model(
                                        ts_df, ts_column, target, forecast_period,
                                        score_type, verbose, conf_int)
            ml_dict[name]['model'] = model
            ml_dict[name]['forecast'] = forecast_df['yhat'].values
            ##### Make sure that RMSE works, if not set it to np.inf  #########
            if score_type == 'rmse':
                score_val = rmse
            else:
                score_val = norm_rmse
        except:
            print('    FB Prophet may not be installed or Model is not running...')
            score_val = np.inf
        ml_dict[name][score_type] = score_val
    elif model_type.lower() == 'stats' or model_type.lower() == 'all':
        ##### First let's try the following models in sequence #########################################
        nsims = 100  ### this is needed only for M-H models in PyFlux
        name = 'PyFlux'
        print(colorful.BOLD + '\nRunning PyFlux Model...' + colorful.END)
        try:
            ml_dict[name]['model'], ml_dict[name]['forecast'], rmse, norm_rmse = build_pyflux_model(ts_df,target,p_max,
                                                        q_max,d_max,
                                            forecast_period, 'MLE', nsims, score_type,verbose)
        except:
            print('    PyFlux model error: predictions not available.')
            score_val = np.inf
            rmse = np.inf
            norm_rmse = np.inf
        ##### Make sure that RMSE works, if not set it to np.inf  #########
        if score_type == 'rmse':
            score_val = rmse
        else:
            score_val = norm_rmse
        ml_dict[name][score_type] = score_val
        ################### Let's build an ARIMA Model and add results #################
        name = 'ARIMA'
        print(colorful.BOLD + '\nRunning Non Seasonal ARIMA Model...' + colorful.END)
        try:
            ml_dict[name]['model'], ml_dict[name]['forecast'], rmse, norm_rmse = build_ARIMA_model(ts_df[target],
                                                    stats_scoring,p_max,d_max,q_max,
                                    forecast_period=forecast_period,method='mle',verbose=verbose)
        except:
            print('    ARIMA model error: predictions not available.')
            score_val = np.inf
        if score_type == 'rmse':
            score_val = rmse
        else:
            score_val = norm_rmse
        ml_dict[name][score_type] = score_val
        ############# Let's build a SARIMAX Model and get results ########################
        name = 'SARIMAX'
        print(colorful.BOLD + '\nRunning Seasonal SARIMAX Model...' + colorful.END)
        try:
            ml_dict[name]['model'], ml_dict[name]['forecast'], rmse, norm_rmse = build_best_SARIMAX_model(ts_df[target], stats_scoring, seasonality,
                                                    seasonal_period, p_max, d_max, q_max,
                                                    forecast_period,verbose)
        except:
            print('    SARIMAX model error: predictions not available.')
            score_val = np.inf
        if score_type == 'rmse':
            score_val = rmse
        else:
            score_val = norm_rmse
        ml_dict[name][score_type] = score_val
        ########### Let's build a VAR Model - but first we have to shift the predictor vars ####
        name = 'VAR'
        if len(preds) == 0:
            print('No VAR model since number of predictors is zero')
            rmse = np.inf
            norm_rmse = np.inf
        else:
            try:
                if df_orig.shape[1] > 1:
                    preds = [x for x in list(df_orig) if x not in [target]]
                    print(colorful.BOLD + '\nRunning VAR Model...' + colorful.END)
                    print('    Shifting %d predictors by 1 to align prior predictor values with current target values...'
                                            %len(preds))
                    ts_df[preds] = ts_df[preds].shift(1)
                    ts_df.dropna(axis=0,inplace=True)
                    ml_dict[name]['model'], ml_dict[name]['forecast'], rmse, norm_rmse = build_VAR_model(ts_df[[target]+preds],stats_scoring,
                                                forecast_period, p_max, q_max)
                else:
                    print(colorful.BOLD + '\nNo predictors available. Skipping VAR model...' + colorful.END)
                    score_val = np.inf
            except:
                print('    VAR model error: predictions not available.')
                rmse = np.inf
                norm_rmse = np.inf
        ################################################################
        if score_type == 'rmse':
            score_val = rmse
        else:
            score_val = norm_rmse
        ########################################################################
        ml_dict[name][score_type] = score_val
        ########## Let's build a Machine Learning Model now with Time Series Data ################
    elif model_type.lower() == 'ml' or model_type.lower() == 'all':
        name = 'ML'
        if len(preds) == 0:
            print('No ML model since number of predictors is zero')
            rmse = np.inf
            norm_rmse = np.inf
        else:
            try:
                if df_orig.shape[1] > 1:
                    preds = [x for x in list(df_orig) if x not in [target]]
                    print(colorful.BOLD + '\nRunning Machine Learning Models...' + colorful.END)
                    print('    Shifting %d predictors by lag=%d to align prior predictor with current target...'
                                %(len(preds),lag))
                    dfxs,target,preds = convert_timeseries_dataframe_to_supervised(ts_df[preds+[target]],
                                            preds+[target],target,n_in=lag,n_out=0,dropT=False)
                    train = dfxs[:-forecast_period]
                    test = dfxs[-forecast_period:]
                    best = quick_ML_model(train[preds],train[target])
                    ml_dict[name]['model'] = best
                    best.set_params(random_state=0)
                    ml_dict[name]['forecast'] = best.fit(train[preds],train[target]).predict(test[preds])
                    rmse, norm_rmse = print_dynamic_rmse(test[target].values,
                                                best.predict(test[preds]),
                                                train[target].values)
                    #### Plotting actual vs predicted for RF Model #################
                    plt.figure(figsize=(5,5))
                    plt.scatter(train.append(test)[target].values,
                                np.r_[best.predict(train[preds]),best.predict(test[preds])])
                    plt.xlabel('Actual')
                    plt.ylabel('Predicted')
                    plt.show()
                else:
                    print(colorful.BOLD + '\nNo predictors available. Skipping Machine Learning model...' + colorful.END)
                    score_val = np.inf
            except:
                print('    For ML model, evaluation score is not available.')
                score_val = np.inf
        ################################################################
        if score_type == 'rmse':
            score_val = rmse
        else:
            score_val = norm_rmse
            rmse = np.inf
            norm_rmse = np.inf
        ########################################################################
        ml_dict[name][score_type] = score_val
        ############ Draw a plot of the Time Series data given so you can select p,d,q ######
        time_series_plot(ts_df[target],chart_time=timeinterval)
    else:
        print('No model_type given or it is unknown type. Please look at input and run again')
        return ml_dict
    ######## Selecting the best model based on the lowest rmse score ######
    f1_stats = {}
    for key, val in ml_dict.items():
        f1_stats[key] = ml_dict[key][score_type]
    best_model_name = min(f1_stats.items(), key=operator.itemgetter(1))[0]
    print(colorful.BOLD + '\nBest Model is:' + colorful.END)
    print('    %s' %best_model_name)
    best_model = ml_dict[best_model_name]['model']
    #print('    Best Model Forecasts: %s' %ml_dict[best_model_name]['forecast'])
    print('    Best Model Score: %0.2f' %ml_dict[best_model_name][score_type])
    return ml_dict
##########################################################
#Defining AUTO_TIMESERIES here
##########################################################
if	__name__	== "__main__":
    version_number = '0.0.10'
    print("""Running Auto Timeseries version: %s...Call by using Auto_Timeseries(trainfile, ts_column,
                            sep=',', target=None, score_type='rmse', forecast_period=2,
                            timeinterval='Month', non_seasonal_pdq=None, seasonality=False,
                            seasonal_period=12, seasonal_PDQ=None,
                            verbose=0)
    To get detailed charts of actuals and forecasts, set verbose = 1""" %version_number)
else:
    version_number = '0.0.10'
    print("""Imported Auto_Timeseries version: %s. Call by using Auto_Timeseries(trainfile, ts_column,
                            sep=',', target=None, score_type='rmse', forecast_period=2,
                            timeinterval='Month', non_seasonal_pdq=None, seasonality=False,
                            seasonal_period=12, seasonal_PDQ=None,
                            verbose=0)
    To get detailed charts of actuals and forecasts, set verbose = 1""" %version_number)
