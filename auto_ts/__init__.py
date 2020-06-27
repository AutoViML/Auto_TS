##########################################################
#Defining AUTO_TIMESERIES here
##########################################################
module_type = 'Running' if  __name__ == "__main__" else 'Imported'
version_number = '0.0.22'
# TODO: Fix based on new interface
print("""Running Auto Timeseries version: %s...Call by using:
        auto_ts.Auto_Timeseries(traindata, ts_column,
                            target, sep,  score_type='rmse', forecast_period=5,
                            time_interval='Month', non_seasonal_pdq=None, seasonality=False,
                            seasonal_period=12, seasonal_PDQ=None, model_type='stats',
                            verbose=1)
    To run three models from Stats, ML and FB Prophet, set model_type='best'""" % version_number)
print("To remove previous versions, perform 'pip uninstall auto_ts'")
print('To get the latest version, perform "pip install auto_ts --no-cache-dir --ignore-installed"')




import warnings
from typing import List, Dict, Optional, Tuple, Union

from datetime import datetime
import copy
import pdb
from collections import defaultdict
import operator
import time

# Tabular Data 
import pandas as pd  # type: ignore
import numpy as np  # type: ignore

# Modeling
from sklearn.exceptions import DataConversionWarning # type: ignore
#### The warnings from Sklearn are so annoying that I have to shut it off ####
warnings.filterwarnings("ignore")
warnings.filterwarnings(action='ignore', category=DataConversionWarning)


def warn(*args, **kwargs):
    pass
warnings.warn = warn

############################################################
import seaborn as sns  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
sns.set(style="white", color_codes=True)

#######################################
# Models
from .models import build_pyflux_model
from .models import BuildArima, BuildSarimax, BuildVAR, BuildML
from .models.build_prophet import BuildProphet


# Utils
from .utils import colorful, load_ts_data, convert_timeseries_dataframe_to_supervised, \
                   time_series_plot, print_static_rmse, print_dynamic_rmse


class AutoTimeSeries:
    def __init__(
        self,
        forecast_period: int, 
        score_type: str = 'rmse',
        time_interval: str = '',
        non_seasonal_pdq: Optional[Tuple]=None,
        seasonality: bool = False,
        seasonal_period: int = 12,
        seasonal_PDQ: Optional[Tuple]=None,
        conf_int: float = 0.95,
        model_type: Union[str, List] ="stats",
        verbose: int = 0
    ):
        """
        Initializae an AutoTimeSeries object
        # TODO: Add complete docstring
        # TODO: Add object types

        :param: non_seasonal_pdq Indicates the maximum value of p, d, q to be used in the search for the best models.
        If None, then the following values are assumed max_p = 3, max_d = 1, max_q = 3
        :type non_seasonal_pdq Optional[Tuple]

        :param seasonality Used in the building of the SARIMAX model only at this time
        :type seasonality bool

        TODO: seasonal_period is calculated internally, so why is it needed to be passed
        :param seasonal_period: Check if this needs to be passed at all.

        TODO: seasonal_PDQ is not being used anywhere. Maybe this is a placeholder to be used in the future.
        :param seasonal_PDQ: Check if this needs to be passed at all.




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

        self.ml_dict: Dict = {}
        self.score_type: str = score_type
        self.forecast_period = forecast_period
        self.time_interval = time_interval
        self.non_seasonal_pdq = non_seasonal_pdq
        self.seasonality = seasonality
        self.seasonal_period = seasonal_period
        self.seasonal_PDQ = seasonal_PDQ  # TODO: This is not being used anywhere. Check if this is needed.
        self.conf_int = conf_int
        if isinstance(model_type, str):
            model_type = [model_type]
        self.model_type = model_type
        self.verbose = verbose

        self.allowed_models = ['best', 'prophet', 'pyflux', 'stats', 'ARIMA', 'SARIMAX', 'VAR', 'ML']

    def fit(
        self,
        traindata: Union[str, pd.DataFrame],
        ts_column: Union[str, int, List[str]],
        target: Union[str, List[str]],
        sep: str = ','):
        """
        Train the AutoTimeseries object
        # TODO: Complete docstring
        """

        print("Start of Fit.....")


        ##### Best hyper-parameters in statsmodels chosen using the best aic, bic or whatever. Select here.
        stats_scoring = 'aic'
        # seed = 99  # Unused

        ### If run_prophet is set to True, then only 1 model will be run and that is FB Prophet ##
        lag = copy.deepcopy(self.forecast_period)-1
        if type(self.non_seasonal_pdq) == tuple:
            p_max = self.non_seasonal_pdq[0]
            d_max = self.non_seasonal_pdq[1]
            q_max = self.non_seasonal_pdq[2]
        else:
            p_max = 3
            d_max = 1
            q_max = 3
        
        # Check 'ts_column' type
        if isinstance(ts_column, int):
            ### If ts_column is a number, then it means you need to convert it to a named variable
            ts_column = list(ts_df)[ts_column]
        if isinstance(ts_column, list):
            # If it is of type List, just pick the first one
            print("\nYou have provided a list as the 'ts_column' argument. Will pick the first value as the 'ts_column' name.")
            ts_column = ts_column[0]
        
        # Check 'target' type
        if isinstance(target, list):
            target = target[0]
            print('    Taking the first column in target list as Target variable = %s' %target)
        else:
            print('    Target variable = %s' %target)

        print("Start of loading of data.....")
        ########## This is where we start the loading of the data file ######################
        if isinstance(traindata, str):
            if traindata != '':
                try:
                    ts_df = load_ts_data(traindata, ts_column, sep, target)
                    if isinstance(ts_df, str):
                        print("""Time Series column %s could not be converted to a Pandas date time column.
                            Please convert your input into a date-time column  and try again""" %ts_column)
                        return None
                    else:
                        print('    File loaded successfully. Shape of data set = %s' %(ts_df.shape,))
                except:
                    print('File could not be loaded. Check the path or filename and try again')
                    return None
        elif isinstance(traindata, pd.DataFrame):
            print('Input is data frame. Performing Time Series Analysis')
            ts_df = load_ts_data(traindata, ts_column, sep, target)
            if isinstance(ts_df, str):
                print("""Time Series column %s could not be converted to a Pandas date time column.
                    Please convert your input into a date-time column  and try again""" %ts_column)
                return None
            else: 
                print('    Dataframe loaded successfully. Shape of data set = %s' %(ts_df.shape,))
        else:
            print('File name is an empty string. Please check your input and try again')
            return None
        df_orig = copy.deepcopy(ts_df)
        if ts_df.shape[1] == 1:
            ### If there is only one column, you assume that to be the target column ####
            target = list(ts_df)[0]
                
                
        preds = [x for x in list(ts_df) if x not in [ts_column, target]]

        ##################################################################################################
        ### Turn the time series index into a variable and calculate the difference.
        ### If the difference is not in days, then it is a hourly or minute based time series
        ### If the difference a multiple of days, then test it for weekly, monthly, qtrly, annual etc.
        ##################################################################################################
        if ts_df.index.dtype=='int' or ts_df.index.dtype=='float':
            ### You must convert the ts_df index into a date-time series using the ts_column given ####
            ts_df = ts_df.set_index(ts_column)
        ts_index = ts_df.index

        ## TODO: Be sure to also assign a frequency to the index column
        ## This will be helpful when finding the "future dataframe" especially for ARIMA, and ML.
        # https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases

        print("Start of figuring out the time interval.....")
        ##################    IF TIME INTERVAL IS NOT GIVEN DO THIS   ########################
        #### This is where the program tries to tease out the time period in the data set ####
        ######################################################################################
        if self.time_interval == '':
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
                    self.time_interval = 'weeks'
                elif diff_in_days == 1:
                    print('it is a Daily time series.')
                    self.time_interval = 'days'
                elif 28 <= diff_in_days < 89:
                    print('it is a Monthly time series.')
                    self.time_interval = 'months'
                elif 89 <= diff_in_days < 178:
                    print('it is a Quarterly time series.')
                    self.time_interval = 'qtr'
                elif 178 <= diff_in_days < 360:
                    print('it is a Semi Annual time series.')
                    self.time_interval = 'qtr'
                elif diff_in_days >= 360:
                    print('it is an Annual time series.')
                    self.time_interval = 'years'
                else:
                    print('Time Series time delta is unknown')
                    return None
            if diff_in_days == 0:
                if diff_in_hours == 0:
                    print('Time series input in Minutes or Seconds = %s' % diff_in_hours)
                    print('it is a Minute time series.')
                    self.time_interval = 'minutes'
                elif diff_in_hours >= 1:
                    print('it is an Hourly time series.')
                    self.time_interval = 'hours'
                else:
                    print('It is an Unknown Time Series delta')
                    return None
        else:
            print('Time Interval is given as %s' % self.time_interval)

        ################# This is where you test the data and find the time interval #######
        self.time_interval = self.time_interval.strip().lower()
        if self.time_interval in ['months', 'month', 'm']:
            self.time_interval = 'months'
            self.seasonal_period = 12
        elif self.time_interval in ['days', 'daily', 'd']:
            self.time_interval = 'days'
            self.seasonal_period = 30
            # Commented out b/c resample only works with DatetimeIndex, not Index
            # ts_df = ts_df.resample('D').sum()
        elif self.time_interval in ['weeks', 'weekly', 'w']:
            self.time_interval = 'weeks'
            self.seasonal_period = 52
        elif self.time_interval in ['qtr', 'quarter', 'q']:
            self.time_interval = 'qtr'
            self.seasonal_period = 4
        elif self.time_interval in ['years', 'year', 'annual', 'y', 'a']:
            self.time_interval = 'years'
            self.seasonal_period = 1
        elif self.time_interval in ['hours', 'hourly', 'h']:
            self.time_interval = 'hours'
            self.seasonal_period = 24
        elif self.time_interval in ['minutes', 'minute', 'min', 'n']:
            self.time_interval = 'minutes'
            self.seasonal_period = 60
        elif self.time_interval in ['seconds', 'second', 'sec', 's']:
            self.time_interval = 'seconds'
            self.seasonal_period = 60
        else:
            self.time_interval = 'months'
            self.seasonal_period = 12

        ########################### This is where we store all models in a nested dictionary ##########
        mldict = lambda: defaultdict(mldict)
        self.ml_dict = mldict()
        try:
            if self.__any_contained_in_list(what_list=['best'], in_list=self.model_type):
                print(colorful.BOLD +'WARNING: Running best models will take time... Be Patient...' + colorful.END)
        except:
            print('Check if your model type is a string or one of the available types of models')
        

        print("Start of Prophet.....")

        ######### This is when you need to use FB Prophet ###################################
        ### When the time interval given does not match the tested_time_interval, then use FB.
        #### Also when the number of rows in data set is very large, use FB Prophet, It is fast.
        #########                 FB Prophet              ###################################

        if self.__any_contained_in_list(what_list=['prophet', 'best'], in_list=self.model_type):
            print("\n")
            print("="*50)
            print("Building Prophet Model")
            print("="*50)
            print("\n")

            name = 'FB_Prophet'
            # Placeholder for cases when model can not be built
            score_val = np.inf 
            model_build = None 
            model = None
            forecasts = None
            print(colorful.BOLD + '\nRunning Facebook Prophet Model...' + colorful.END)
            try:
                #### If FB prophet needs to run, it needs to be installed. Check it here ###
                model_build = BuildProphet(self.forecast_period, self.time_interval,
                                            self.score_type, self.verbose, self.conf_int)
                model, forecast_df, rmse, norm_rmse = model_build.fit(
                                            ts_df, ts_column, target)

                forecasts = forecast_df['yhat'].values
                
                ##### Make sure that RMSE works, if not set it to np.inf  #########
                if self.score_type == 'rmse':
                    score_val = rmse
                else:
                    score_val = norm_rmse
            except Exception as e:  
                print("Exception occured while building Prophet model...")
                print(e)
                print('    FB Prophet may not be installed or Model is not running...')
                
            self.ml_dict[name]['model'] = model
            self.ml_dict[name]['forecast'] = forecasts
            self.ml_dict[name][self.score_type] = score_val
            self.ml_dict[name]['model_build'] = model_build
        

        print("Start of Stats.....")
        if self.__any_contained_in_list(what_list=['pyflux', 'stats', 'best'], in_list=self.model_type):
            print("\n")
            print("="*50)
            print("Building PyFlux Model")
            print("="*50)
            print("\n")


            print("Start of PyFlux.....")
            ##### First let's try the following models in sequence #########################################
            nsims = 100   ### this is needed only for M-H models in PyFlux
            name = 'PyFlux'
            # Placeholder for cases when model can not be built
            score_val = np.inf 
            model_build = None 
            model = None
            forecasts = None
            print(colorful.BOLD + '\nRunning PyFlux Model...' + colorful.END)
            try:
                model, forecasts, rmse, norm_rmse = \
                    build_pyflux_model(ts_df, target, p_max, q_max, d_max, self.forecast_period,
                                    'MLE', nsims, self.score_type, self.verbose)

                ##### Make sure that RMSE works, if not set it to np.inf  #########
                if isinstance(rmse, str):
                    print('    PyFlux not installed. Install PyFlux and run it again')
                else:
                    if self.score_type == 'rmse':
                        score_val = rmse
                    else:
                        score_val = norm_rmse                
            except Exception as e:  
                print("Exception occured while building PyFlux model...")
                print(e)
                print('    PyFlux model error: predictions not available.')

            self.ml_dict[name]['model'] = model
            self.ml_dict[name]['forecast'] = forecasts
            self.ml_dict[name][self.score_type] = score_val
            self.ml_dict[name]['model_build'] = model_build  # TODO: Add the right value here

        print("Start of ARIMA.....")
        if self.__any_contained_in_list(what_list=['ARIMA', 'stats', 'best'], in_list=self.model_type): 
            ################### Let's build an ARIMA Model and add results #################
            print("\n")
            print("="*50)
            print("Building ARIMA Model")
            print("="*50)
            print("\n")

            name = 'ARIMA'

            # Placeholder for cases when model can not be built
            score_val = np.inf 
            model_build = None 
            model = None
            forecasts = None
            print(colorful.BOLD + '\nRunning Non Seasonal ARIMA Model...' + colorful.END)
            try:
                model_build = BuildArima(
                    stats_scoring, p_max, d_max, q_max,
                    forecast_period=self.forecast_period, method='mle', verbose=self.verbose
                )
                model, forecasts, rmse, norm_rmse = model_build.fit(ts_df[target])

                if self.score_type == 'rmse':
                    score_val = rmse
                else:
                    score_val = norm_rmse
            except Exception as e:  
                print("Exception occured while building ARIMA model...")
                print(e)
                print('    ARIMA model error: predictions not available.')
                
            self.ml_dict[name]['model'] = model
            self.ml_dict[name]['forecast'] = forecasts
            self.ml_dict[name][self.score_type] = score_val
            self.ml_dict[name]['model_build'] = model_build  
        
        print("Start of SARIMAX.....")
        if self.__any_contained_in_list(what_list=['SARIMAX', 'stats', 'best'], in_list=self.model_type):
            ############# Let's build a SARIMAX Model and get results ########################
            print("\n")
            print("="*50)
            print("Building SARIMAX Model")
            print("="*50)
            print("\n")

            name = 'SARIMAX'
            # Placeholder for cases when model can not be built
            score_val = np.inf 
            model_build = None 
            model = None
            forecasts = None

            print(colorful.BOLD + '\nRunning Seasonal SARIMAX Model...' + colorful.END)
            try:
                model_build = BuildSarimax(
                    scoring=stats_scoring,
                    seasonality=self.seasonality,
                    seasonal_period=self.seasonal_period,
                    p_max=p_max, d_max=d_max, q_max=q_max,
                    forecast_period=self.forecast_period,
                    verbose=self.verbose
                )
                # TODO: https://github.com/AutoViML/Auto_TS/issues/10
                model, forecasts, rmse, norm_rmse = model_build.fit(
                    ts_df=ts_df[[target]+preds],  
                    target_col=target                    
                )

                if self.score_type == 'rmse':
                    score_val = rmse
                else:
                    score_val = norm_rmse
            except Exception as e:  
                print("Exception occured while building SARIMAX model...")
                print(e)
                print('    SARIMAX model error: predictions not available.')
                
            self.ml_dict[name]['model'] = model
            self.ml_dict[name]['forecast'] = forecasts
            self.ml_dict[name][self.score_type] = score_val
            self.ml_dict[name]['model_build'] = model_build

        print("Start of VAR.....")
        if self.__any_contained_in_list(what_list=['VAR', 'stats', 'best'], in_list=self.model_type):
            ########### Let's build a VAR Model - but first we have to shift the predictor vars ####

            print("\n")
            print("="*50)
            print("Building VAR Model")
            print("="*50)
            print("\n")

            name = 'VAR'
            # Placeholder for cases when model can not be built
            score_val = np.inf 
            model_build = None 
            model = None
            forecasts = None
            
            if len(preds) == 0:
                print(colorful.BOLD + '\nNo VAR model created since no explanatory variables given in data set' + colorful.END)
            else:
                try:
                    print(colorful.BOLD + '\nRunning VAR Model...' + colorful.END)
                    print('    Shifting %d predictors by 1 to align prior predictor values with current target values...'
                                            %len(preds))

                    # TODO: This causes an issue later in ML (most likely cause of https://github.com/AutoViML/Auto_TS/issues/15)
                    # Since we are passing ts_df there. Make sure you dont assign it 
                    # back to the same variable. Make a copy and make changes to that copy.
                    ts_df_shifted = ts_df.copy(deep=True)
                    ts_df_shifted[preds] = ts_df_shifted[preds].shift(1)
                    ts_df_shifted.dropna(axis=0,inplace=True)

                    model_build = BuildVAR(criteria=stats_scoring, forecast_period=self.forecast_period, p_max=p_max, q_max=q_max)
                    model, forecasts, rmse, norm_rmse = model_build.fit(
                        ts_df_shifted[[target]+preds])

                    if self.score_type == 'rmse':
                        score_val = rmse
                    else:
                        score_val = norm_rmse
                except Exception as e:  
                    print("Exception occured while building VAR model...")
                    print(e)
                    warnings.warn('    VAR model error: predictions not available.')
                    
            self.ml_dict[name]['model'] = model
            self.ml_dict[name]['forecast'] = forecasts
            self.ml_dict[name][self.score_type] = score_val
            self.ml_dict[name]['model_build'] = model_build  
        
        print("Start of ML.....")
        if self.__any_contained_in_list(what_list=['ml', 'best'], in_list=self.model_type):
            ########## Let's build a Machine Learning Model now with Time Series Data ################
            
            print("\n")
            print("="*50)
            print("Building ML Model")
            print("="*50)
            print("\n")
            
            name = 'ML'
            # Placeholder for cases when model can not be built
            score_val = np.inf 
            model_build = None 
            model = None
            forecasts = None
            
            if len(preds) == 0:
                print(colorful.BOLD + '\nNo predictors available. Skipping Machine Learning model...' + colorful.END)
            else:
                try:
                    print(colorful.BOLD + '\nRunning Machine Learning Models...' + colorful.END)
                    print('    Shifting %d predictors by lag=%d to align prior predictor with current target...'
                                % (len(preds), lag))
            
                    model_build = BuildML(
                        scoring=self.score_type,
                        forecast_period = self.forecast_period,
                        verbose=self.verbose
                    )
                    
                    # best = model_build.fit(ts_df=ts_df, target_col=target, lags=lag)
                    model, forecasts, rmse, norm_rmse = model_build.fit(
                        ts_df=ts_df,
                        target_col=target,
                        lags=lag
                    )

                    if self.score_type == 'rmse':
                        score_val = rmse
                    else:
                        score_val = norm_rmse
                    # bestmodel = best[0]
                                            
                    # #### Plotting actual vs predicted for ML Model #################
                    # # TODO: Move inside the Build Class
                    # plt.figure(figsize=(5, 5))
                    # plt.scatter(train.append(test)[target].values,
                    #             np.r_[bestmodel.predict(train[preds]), bestmodel.predict(test[preds])])
                    # plt.xlabel('Actual')
                    # plt.ylabel('Predicted')
                    # plt.show(block=False)
                    # ############ Draw a plot of the Time Series data ######
                    # time_series_plot(dfxs[target], chart_time=self.time_interval)
                                        
                except Exception as e:  
                    print("Exception occured while building ML model...")
                    print(e)
                    print('    For ML model, evaluation score is not available.')
  
            self.ml_dict[name]['model'] = model
            self.ml_dict[name]['forecast'] = forecasts
            self.ml_dict[name][self.score_type] = score_val
            self.ml_dict[name]['model_build'] = model_build  
            
        if not self.__all_contained_in_list(what_list=self.model_type, in_list=self.allowed_models):
            print(f'The model_type should be any of the following: {self.allowed_models}. You entered {self.model_type}. Some models may not have been developed...')
            if len(list(self.ml_dict.keys())) == 0:  
                return None        
        
        ######## Selecting the best model based on the lowest rmse score ######
        best_model_name = self.get_best_model_name()    
        print(colorful.BOLD + '\nBest Model is:' + colorful.END)
        print('    %s' % best_model_name)
        
        print('    Best Model Forecasts: %s' %self.ml_dict[best_model_name]['forecast'])
        print('    Best Model Score: %0.2f' % self.ml_dict[best_model_name][self.score_type])
        return self

        

    def get_best_model_name(self) -> str:
        """
        Returns the best model name
        """
        f1_stats = {}
        for key, _ in self.ml_dict.items():
            f1_stats[key] = self.ml_dict[key][self.score_type]
        best_model_name = min(f1_stats.items(), key=operator.itemgetter(1))[0]
        return best_model_name

    def get_best_model(self):
        """
        Returns the best model after training
        """
        return self.ml_dict.get(self.get_best_model_name()).get('model')

    def get_model(self, model_name: str):
        """
        Returns the specified model
        """
        if self.ml_dict.get(model_name) is not None:
            return self.ml_dict.get(model_name).get('model')
        else:
            print(f"Model with name '{model_name}' does not exist.")
            return None

    def get_best_model_build(self):
        """
        Returns the best model after training
        """
        return self.ml_dict.get(self.get_best_model_name()).get('model_build')

    def get_model_build(self, model_name: str):
        """
        Returns the specified model
        """
        if self.ml_dict.get(model_name) is not None:
            return self.ml_dict.get(model_name).get('model_build')
        else:
            print(f"Model with name '{model_name}' does not exist.")
            return None
    


    def get_ml_dict(self):
        """
        Returns the entire ML Dictionary
        """
        return self.ml_dict

    def predict(
        self,
        model: str = 'best',
        X_exogen: Optional[pd.DataFrame]=None,
        forecast_period: Optional[int] = None,
        simple: bool = True) -> Optional[np.array]:
        """
        Predict the results
        """
        if model.lower() == 'best': 
            predictions = self.get_best_model_build().predict(
                X_exogen = X_exogen,
                forecast_period=forecast_period,
                simple=simple
            )
        elif self.get_model_build(model) is not None:
            predictions = self.get_model_build(model).predict(
                X_exogen = X_exogen,
                forecast_period=forecast_period,
                simple=simple
            )
        else:
            warnings.warn(f"Model of type '{model}' does not exist. No predictions will be made.")
            predictions = None 

        return predictions

    def get_leaderboard(self, ascending=True) -> pd.DataFrame:
        """
        Returns the leaderboard after fitting
        """
        names = []
        rmses = []
        for model_name in list(self.ml_dict.keys()):
            names.append(model_name)
            rmses.append(self.ml_dict.get(model_name).get(self.score_type))
            
        results = pd.DataFrame({"name": names, self.score_type: rmses})
        results.sort_values(self.score_type, ascending=ascending, inplace=True)
        return results

    def __any_contained_in_list(self, what_list: List[str], in_list: List[str], lower: bool = True) -> bool:
        """
        Returns True is any element in the 'in_list' is contained in the 'what_list'
        """
        if lower:
            what_list = [elem.lower() for elem in what_list]
            in_list = [elem.lower() for elem in in_list]

        return any([True if elem in in_list else False for elem in what_list])

    def __all_contained_in_list(self, what_list: List[str], in_list: List[str], lower: bool = True) -> bool:
        """
        Returns True is all elements in the 'in_list' are contained in the 'what_list'
        """
        if lower:
            what_list = [elem.lower() for elem in what_list]
            in_list = [elem.lower() for elem in in_list]

        return all([True if elem in in_list else False for elem in what_list])



