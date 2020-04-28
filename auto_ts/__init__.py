####################################################################################
import pandas as pd
import numpy as np
from datetime import datetime
#############################################################
import warnings
warnings.filterwarnings("ignore")
from sklearn.exceptions import DataConversionWarning
warnings.filterwarnings(action='ignore', category=DataConversionWarning)
#### The warnings from Sklearn are so annoying that I have to shut it off ####
import warnings
warnings.filterwarnings("ignore")
def warn(*args, **kwargs):
    pass
warnings.warn = warn
############################################################
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
sns.set(style="white", color_codes=True)
import copy
import pdb
from collections import defaultdict
import operator
import time
#######################################
# Models
from .models import build_arima_model, build_sarimax_model, build_var_model, \
                    build_pyflux_model, build_prophet_model, run_ensemble_model
#from .models import build_ml, build_prophet, build_pyflux

# Utils
from .utils import colorful, load_ts_data, convert_timeseries_dataframe_to_supervised, \
                   time_series_plot, print_static_rmse, print_dynamic_rmse
#from .utils import colors, eda, etl, metrics, val


def Auto_Timeseries(traindata, ts_column, target, sep=',', score_type='rmse',
                    forecast_period=5, time_interval='', non_seasonal_pdq=None,
                    seasonality=False, seasonal_period=12, seasonal_PDQ=None,
                    conf_int=0.95, model_type="stats", verbose=0):
    """
    ####################################################################################
    ####                          Auto Time Series                                  ####
    ####                           Version 0.0.19 Version                           ####
    ####                    Conceived and Developed by Ram Seshadri                 ####
    ####                        All Rights Reserved                                 ####
    ####################################################################################
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
    time_interval: default is "Month". What is the time period in your data set. Options are: "days",
    model_type: default is "stats". Choice is between "stats", "prophet" and "ml". "All" will build all.
        - "stats" will build statsmodels based ARIMA< SARIMAX and VAR models
        - "ml" will build a machine learning model using Random Forests provided explanatory vars are given
        - "prophet" will build a model using FB Prophet -> this means you must have FB Prophet installed
        - "best" will build three of the best models from above which might take some time for large data sets.
    We recommend that you choose a small sample from your data set bedfore attempting to run entire data.
    #####################################################################################################
    and the evaluation metric so it can select the best model. Currently only 2 are supported: RMSE and
    Normalized RMSE (ratio of RMSE to the standard deviation of actuals). Other eval metrics will be soon.
    the target variable you are trying to predict (if there is more than one variable in your data set),
    and the time interval that is in the data. If your data is in a different time interval than given,
    Auto_Timeseries will automatically resample your data to the given time interval and learn to make
    predictions. Notice that except for filename and ts_column which are required, all others are optional.
    Note that optionally you can give a separator for the data in your file. Default is comman (",").
    "time_interval" options are: 'Days', 'Weeks', 'Months', 'Qtr', 'Year', 'Minutes', 'Hours', 'Seconds'.
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
        p_max = 3
        d_max = 1
        q_max = 3
    ################################
    if type(seasonal_PDQ) == tuple:
        seasonal_order = copy.deepcopy(seasonal_PDQ)
    else:
        seasonal_order = (3, 1, 3)

    ########## This is where we start the loading of the data file ######################
    if isinstance(traindata, str):
        if traindata != '':
            try:
                ts_df = load_ts_data(traindata, ts_column, sep, target)
                if isinstance(ts_df, str):
                    print("""Time Series column %s could not be converted to a Pandas date time column.
                         Please convert your input into a date-time column  and try again""" %ts_column)
                    return
                else:
                    print('    File loaded successfully. Shape of data set = %s' %(ts_df.shape,))
            except:
                print('File could not be loaded. Check the path or filename and try again')
                return
    elif isinstance(traindata, pd.DataFrame):
        print('Input is data frame. Performing Time Series Analysis')
        ts_df = load_ts_data(traindata, ts_column, sep, target)
        if isinstance(ts_df, str):
            print("""Time Series column %s could not be converted to a Pandas date time column.
                 Please convert your input into a date-time column  and try again""" %ts_column)
            return
        else:
            print('    Dataframe loaded successfully. Shape of data set = %s' %(ts_df.shape,))
    else:
        print('File name is an empty string. Please check your input and try again')
        return
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
    if time_interval == '':
        ts_index = pd.to_datetime(ts_df.index)
        diff = (ts_index[1] - ts_index[0]).to_pytimedelta()
        diffdays = diff.days
        diffsecs = diff.seconds
        if diffsecs == 0:
            diff_in_hours = 0
            diff_in_days = abs(diffdays)
        else:
            diff_in_hours = abs(diffdays*24*3600 + diffsecs)/3600
        if diff_in_hours == 0 and diff_in_days >= 1:
            print('Time series input in days = %s' % diff_in_days)
            if diff_in_days == 7:
                print('it is a Weekly time series.')
                time_interval = 'weeks'
            elif diff_in_days == 1:
                print('it is a Daily time series.')
                time_interval = 'days'
            elif 28 <= diff_in_days < 89:
                print('it is a Monthly time series.')
                time_interval = 'months'
            elif 89 <= diff_in_days < 178:
                print('it is a Quarterly time series.')
                time_interval = 'qtr'
            elif 178 <= diff_in_days < 360:
                print('it is a Semi Annual time series.')
                time_interval = 'qtr'
            elif diff_in_days >= 360:
                print('it is an Annual time series.')
                time_interval = 'years'
            else:
                print('Time Series time delta is unknown')
                return
        if diff_in_days == 0:
            if diff_in_hours == 0:
                print('Time series input in Minutes or Seconds = %s' % diff_in_hours)
                print('it is a Minute time series.')
                time_interval = 'minutes'
            elif diff_in_hours >= 1:
                print('it is an Hourly time series.')
                time_interval = 'hours'
            else:
                print('It is an Unknown Time Series delta')
                return
    else:
        print('Time Interval is given as %s' % time_interval)

    ################# This is where you test the data and find the time interval #######
    time_interval = time_interval.strip().lower()
    if time_interval in ['months', 'month', 'm']:
        time_interval = 'months'
        seasonal_period = 12
    elif time_interval in ['days', 'daily', 'd']:
        time_interval = 'days'
        seasonal_period = 30
        # Commented out b/c resample only works with DatetimeIndex, not Index
        # ts_df = ts_df.resample('D').sum()
    elif time_interval in ['weeks', 'weekly', 'w']:
        time_interval = 'weeks'
        seasonal_period = 52
    elif time_interval in ['qtr', 'quarter', 'q']:
        time_interval = 'qtr'
        seasonal_period = 4
    elif time_interval in ['years', 'year', 'annual', 'y', 'a']:
        time_interval = 'years'
        seasonal_period = 1
    elif time_interval in ['hours', 'hourly', 'h']:
        time_interval = 'hours'
        seasonal_period = 24
    elif time_interval in ['minutes', 'minute', 'min', 'n']:
        time_interval = 'minutes'
        seasonal_period = 60
    elif time_interval in ['seconds', 'second', 'sec', 's']:
        time_interval = 'seconds'
        seasonal_period = 60
    else:
        time_interval = 'months'
        seasonal_period = 12

    ########################### This is where we store all models in a nested dictionary ##########
    mldict = lambda: defaultdict(mldict)
    ml_dict = mldict()
    try:
        if model_type.lower() == 'best':
            print(colorful.BOLD +'WARNING: Running best models will take time... Be Patient...' + colorful.END)
    except:
        print('Check if your model type is a string or one of the available types of models')
    ######### This is when you need to use FB Prophet ###################################
    ### When the time interval given does not match the tested_time_interval, then use FB.
    #### Also when the number of rows in data set is very large, use FB Prophet, It is fast.
    #########                 FB Prophet              ###################################
    if model_type.lower() in ['prophet','best']:
        name = 'FB_Prophet'
        print(colorful.BOLD + '\nRunning Facebook Prophet Model...' + colorful.END)
        # try:
        #### If FB prophet needs to run, it needs to be installed. Check it here ###
        model, forecast_df, rmse, norm_rmse = build_prophet_model(
                                    ts_df, ts_column, target, forecast_period, time_interval,
                                    score_type, verbose, conf_int)
        ml_dict[name]['model'] = model
        ml_dict[name]['forecast'] = forecast_df['yhat'].values
        ##### Make sure that RMSE works, if not set it to np.inf  #########
        if score_type == 'rmse':
            score_val = rmse
        else:
            score_val = norm_rmse
        # except:
        #     print('    FB Prophet may not be installed or Model is not running...')
        #     score_val = np.inf
        ml_dict[name][score_type] = score_val
    if model_type.lower() in ['stats','best']:
        ##### First let's try the following models in sequence #########################################
        nsims = 100   ### this is needed only for M-H models in PyFlux
        name = 'PyFlux'
        print(colorful.BOLD + '\nRunning PyFlux Model...' + colorful.END)
        try:
            ml_dict[name]['model'], ml_dict[name]['forecast'], rmse, norm_rmse = \
                build_pyflux_model(ts_df, target, p_max, q_max, d_max, forecast_period,
                                   'MLE', nsims, score_type, verbose)
            if isinstance(rmse,str):
                print('    PyFlux not installed. Install PyFlux and run it again')
                score_val = np.inf
                rmse = np.inf
                norm_rmse = np.inf
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
            ml_dict[name]['model'], ml_dict[name]['forecast'], rmse, norm_rmse = build_arima_model(ts_df[target],
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
        # try:
        ml_dict[name]['model'], ml_dict[name]['forecast'], rmse, norm_rmse = build_sarimax_model(ts_df[target], stats_scoring, seasonality,
                                                seasonal_period, p_max, d_max, q_max,
                                                forecast_period,verbose)
        # except:
        #     print('    SARIMAX model error: predictions not available.')
        #     score_val = np.inf
        if score_type == 'rmse':
            score_val = rmse
        else:
            score_val = norm_rmse
        ml_dict[name][score_type] = score_val
        ########### Let's build a VAR Model - but first we have to shift the predictor vars ####
        name = 'VAR'
        if len(preds) == 0:
            print(colorful.BOLD + '\nNo VAR model created since no explanatory variables given in data set' + colorful.END)
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
                    ml_dict[name]['model'], ml_dict[name]['forecast'], rmse, norm_rmse = build_var_model(ts_df[[target]+preds],stats_scoring,
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
    if model_type.lower() in ['ml','best']:
        ########## Let's build a Machine Learning Model now with Time Series Data ################
        name = 'ML'
        if len(preds) == 0:
            print('No ML model since number of predictors is zero')
            rmse = np.inf
            norm_rmse = np.inf
        else:
            try:
                if df_orig.shape[1] > 1:
                    preds = [x for x in list(ts_df) if x not in [target]]
                    print(colorful.BOLD + '\nRunning Machine Learning Models...' + colorful.END)
                    print('    Shifting %d predictors by lag=%d to align prior predictor with current target...'
                                % (len(preds), lag))
                    # ipdb.set_trace()
                    dfxs, target, preds = convert_timeseries_dataframe_to_supervised(ts_df[preds+[target]],
                                            preds+[target], target, n_in=lag, n_out=0, dropT=False)
                    train = dfxs[:-forecast_period]
                    test = dfxs[-forecast_period:]
                    best = run_ensemble_model(train[preds], train[target], 'TimeSeries',
                                              score_type, verbose)
                    bestmodel = best[0]
                    ml_dict[name]['model'] = bestmodel
                    ### Certain models dont have random state => so dont do this for all since it will error
                    #best.set_params(random_state=0)
                    ml_dict[name]['forecast'] = bestmodel.fit(train[preds],train[target]).predict(test[preds])
                    rmse, norm_rmse = print_dynamic_rmse(test[target].values,
                                                bestmodel.predict(test[preds]),
                                                train[target].values)
                    #### Plotting actual vs predicted for RF Model #################
                    plt.figure(figsize=(5, 5))
                    plt.scatter(train.append(test)[target].values,
                                np.r_[bestmodel.predict(train[preds]), bestmodel.predict(test[preds])])
                    plt.xlabel('Actual')
                    plt.ylabel('Predicted')
                    plt.show()
                    ############ Draw a plot of the Time Series data ######
                    time_series_plot(dfxs[target], chart_time=time_interval)
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
    if not model_type.lower() in ['stats','ml', 'prophet', 'best']:
        print('The model_type should be either stats, prophet, ml or best. Check your input and try again...')
        return ml_dict
    ######## Selecting the best model based on the lowest rmse score ######
    f1_stats = {}
    for key, val in ml_dict.items():
        f1_stats[key] = ml_dict[key][score_type]
    best_model_name = min(f1_stats.items(), key=operator.itemgetter(1))[0]
    print(colorful.BOLD + '\nBest Model is:' + colorful.END)
    print('    %s' % best_model_name)
    best_model = ml_dict[best_model_name]['model']
    print('    Best Model Forecasts: %s' %ml_dict[best_model_name]['forecast'])
    print('    Best Model Score: %0.2f' % ml_dict[best_model_name][score_type])
    return ml_dict

##########################################################
#Defining AUTO_TIMESERIES here
##########################################################
module_type = 'Running' if  __name__ == "__main__" else 'Imported'
version_number = '0.0.19'
print("""Running Auto Timeseries version: %s...Call by using:
        auto_ts.Auto_Timeseries(traindata, ts_column,
                            target, sep,  score_type='rmse', forecast_period=5,
                            time_interval='Month', non_seasonal_pdq=None, seasonality=False,
                            seasonal_period=12, seasonal_PDQ=None, model_type='stats',
                            verbose=1)
    To run three models from Stats, ML and FB Prophet, set model_type='best'""" % version_number)
print("To remove previous versions, perform 'pip uninstall auto_ts'")
print('To get the latest version, perform "pip install auto_ts --no-cache-dir --ignore-installed"')
