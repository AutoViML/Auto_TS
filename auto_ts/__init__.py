##########################################################
#Defining AUTO_TIMESERIES here
##########################################################
import warnings
warnings.filterwarnings(action='ignore')
from typing import List, Dict, Optional, Tuple, Union

from datetime import datetime
import copy
from collections import defaultdict
import operator
from time import time
import pdb

# Tabular Data
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import dask
import dask.dataframe as dd

# Modeling
from sklearn.exceptions import DataConversionWarning # type: ignore
#### The warnings from Sklearn are so annoying that I have to shut it off ####
warnings.filterwarnings("ignore")
warnings.filterwarnings(action='ignore', category=DataConversionWarning)

# Plotting
import seaborn as sns  # type: ignore

def warn(*args, **kwargs):
    pass
warnings.warn = warn

############################################################
import seaborn as sns  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
sns.set(style="white", color_codes=True)

#######################################
# Models
from .models import BuildBase, BuildArima, BuildAutoSarimax, BuildVAR, BuildML
from .models.build_prophet import BuildProphet


# Utils
from .utils import colorful, load_ts_data, convert_timeseries_dataframe_to_supervised, \
                   time_series_plot, print_static_rmse, print_dynamic_rmse, quick_ts_plot, \
                   test_stationarity, print_ts_model_stats


class auto_timeseries:
    def __init__(
        self,
        forecast_period: int = 5,
        score_type: str = 'rmse',
        time_interval: Optional[str] = None,
        non_seasonal_pdq: Optional[Tuple]=None,
        seasonality: bool = False,
        seasonal_period: int = 12,
        conf_int: float = 0.95,
        model_type: Union[str, List] = "stats",
        verbose: int = 0,
        *args,
        **kwargs
    ):
        """
        ####################################################################################
        ####                          Auto Time Series                                  ####
        ####         Developed by Ram Seshadri & Expanded by Nikhil Gupta               ####
        ####                        Python 3: 2018-2020                                 ####
        ####################################################################################
        Initialize an auto_timeseries object
        :score_type: The metric used for scoring the models. Default = 'rmse'
        Currently only 2 are supported:
        (1) RMSE
        (2) Normalized RMSE (ratio of RMSE to the standard deviation of actual)
        :type str

        :param time_interval Used to indicate the frequency at which the data is collected
        This is used for two purposes (1) in building the Prophet model and (2) used to impute the seasonal period for SARIMAX in case it is not provided by the user (None). Type is String.
        We use the following codes from Pandas date-range frequency codes link:
        source: "https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases">pandas date range frequency
        These are the same codes that Prophet uses to make the prediction dataframe.

        Hence, please note that these are the list of allowed aliases for time_interval
                              ['B','C','D','W','M','SM','BM','CBM',
                             'MS','SMS','BMS','CBMS','Q','BQ','QS','BQS',
                             'A,Y','BA,BY','AS,YS','BAS,BYS','BH',
                             'H','T,min','S','L,ms','U,us','N']

        For a start, you can test the following codes for your data and see how the results are:

        (1) 'MS', 'M', 'SM', 'BM', 'CBM', 'SMS', 'BMS' for monthly frequency data
        (2) 'D', 'B', 'C' for daily frequency data
        (3) 'W' for weekly frequency data
        (4) 'Q', 'BQ', 'QS', 'BQS' for quarterly frequency data
        (5) 'A,Y', 'BA,BY', 'AS,YS', 'BAS,YAS' for yearly frequency data
        (6) 'BH', 'H', 'h' for hourly frequency data
        (7) 'T,min' for minute frequency data
        (8) 'S', 'L,milliseconds', 'U,microseconds', 'N,nanoseconds' for second frequency data

        Or you can leave it as None and auto_timeseries will try and impute it.
        :type time_interval Optional[str]

        :param: non_seasonal_pdq Indicates the maximum value of p, d, q to be used in the search for "stats" based models.
        If None, then the following values are assumed max_p = 3, max_d = 1, max_q = 3
        :type non_seasonal_pdq Optional[Tuple]

        :param seasonality Used in the building of the SARIMAX model only at this time.
        :type seasonality bool

        :param seasonal_period: Indicates the seasonality period in the data.
        Used in the building of the SARIMAX model only at this time.
        There is no impact of this argument if seasonality is set to False
        If None, the program will try to infer this from the time_interval (frequency) of the data
        We assume the following as defaults but feel free to change them.
        (1) If frequency is Monthly, then seasonal_period is assumed to be 12
        (2) If frequency is Daily, then seasonal_period is assumed to be 30 (but it could be 7)
        (3) If frequency is Weekly, then seasonal_period is assumed to be 52
        (4) If frequency is Quarterly, then seasonal_period is assumed to be 4
        (5) If frequency is Yearly, then seasonal_period is assumed to be 1
        (6) If frequency is Hourly, then seasonal_period is assumed to be 24
        (7) If frequency is Minutes, then seasonal_period is assumed to be 60
        (8) If frequency is Seconds, then seasonal_period is assumed to be 60
        :type seasonal_period int

        :param conf_int: Confidence Interval for building the Prophet model. Default: 0.95
        :type conf_int float

        :param model_type The type(s) of model to build. Default to building only statistical models
        Can be a string or a list of models. Allowed values are:
        'best', 'prophet', 'stats', 'SARIMAX', 'VAR', 'ML'.
        "prophet" will build a model using FB Prophet -> this means you must have FB Prophet installed
        "stats" will build statsmodels based SARIMAX and VAR models
        "ML" will build a machine learning model using Random Forests provided explanatory vars are given
        'best' will try to build all models and pick the best one
        If a list is provided, then only those models will be built
        WARNING: "best" might take some time for large data sets. We recommend that you
        choose a small sample from your data set bedfore attempting to run entire data.
        :type model_type: Union[str, List]

        :param verbose Indicates the verbosity of printing (Default = 0)
        :type verbose int

        ##################################################################################################
        AUTO_TIMESERIES IS A COMPLEX MODEL BUILDING UTILITY FOR TIME SERIES DATA. SINCE IT AUTOMATES MANY
        TASKS INVOLVED IN A COMPLEX ENDEAVOR, IT ASSUMES MANY INTELLIGENT DEFAULTS. BUT YOU CAN CHANGE THEM.
        Auto_Timeseries will rapidly build predictive models based on Statsmodels, Seasonal ARIMA
        and Scikit-Learn ML. It will automatically select the BEST model which gives best score specified.
        #####################################################################################################
        """
        self.ml_dict: Dict = {}
        self.score_type: str = score_type
        self.forecast_period =  forecast_period
        self.time_interval = time_interval
        self.non_seasonal_pdq = non_seasonal_pdq
        self.seasonality = seasonality
        self.seasonal_period = seasonal_period
        self.conf_int = conf_int
        if isinstance(model_type, str):
            model_type = [model_type]
        self.model_type = model_type
        self.verbose = verbose
        self.holidays = None
        self.growth = "linear"
        self.allowed_models = ['best', 'prophet', 'stats', 'ml', 'arima','ARIMA','Prophet','SARIMAX', 'VAR', 'ML']

        # new function.
        if args:
            for each_arg in args:
                print(each_arg)
        if kwargs:
            for key, value in zip(kwargs.keys(), kwargs.values()):
                if key == 'seasonal_PDQ':
                    print('seasonal_PDQ argument is deprecated. Please remove the argument in future.')
                if key == 'holidays':
                    print('holidays argument for FB Prophet given. It must be dictionary or DataFrame.')
                    self.holidays = value
                if key == 'growth':
                    print('growth argument of FB Prophet given. It must be "linear" or "logistic"')
                    self.growth = value

    def fit(
        self,
        traindata: Union[str, pd.DataFrame],
        ts_column: Union[str, int, List[str]],
        target: Union[str, List[str]],
        sep: Optional[str]=',', ## default is comma, string can be anything.
        cv: Optional[int]=5, ### Integer field cannot be defaulted to None
        ):
        """
        Train the auto_timeseries object
        # TODO: Complete docstring

        :param traindata Path for the data file or a dataframe. It accepts both.
        :type traindata Union[str, pd.DataFrame]

        :param ts_column Name of the datetime column in your dataset.
            If it is of type 'str', it will be treated as a column name.
            If it is of type 'int', it will be treated as the column number.
            If it is of type 'List', the first one will be picked and will be treated as the column name.
        :type ts_column Union[str, int, List[str]]

        :param target: Name of the column you are trying to predict. Target could also be the only column in your data

        :type target Union[str, List[str]]
            If it is of type 'str', it will be treated as a column name.
            If it is of type 'List', the first one will be picked and will be treated as the column name.

        :param cv Number of folds to use for cross validation.
            Number of observations in the Validation set for each fold = forecast period
            default is 5 fold cross validation.
        :type cv Optional[int]

        :param sep: Note that optionally you can give a separator for the data in your file.
            Default is None which treats the sep as a comma (datafile as a 'csv').
        :type sep Optional[str]
        """

        list_of_valid_time_ints = ['B','C','D','W','M','SM','BM','CBM',
                                        'MS','SMS','BMS','CBMS','Q','BQ','QS','BQS',
                                        'A,Y','BA,BY','AS,YS','BAS,BYS','BH',
                                        'H','T,min','S','L,ms','U,us','N']

        start = time()
        print("Start of Fit.....")

        ### first test the data for Stationary-ness #############
        if self.verbose >= 1:
            test_stationarity(traindata[target].values, plot=False, verbose=True)

        ##### Best hyper-parameters in statsmodels chosen using the best aic, bic or whatever. Select here.
        stats_scoring = 'aic'

        ### If run_prophet is set to True, then only 1 model will be run and that is FB Prophet ##
        lag = copy.deepcopy(self.forecast_period)-1

        # if type(self.non_seasonal_pdq) == tuple:
        if isinstance(self.non_seasonal_pdq, tuple):
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
            # TODO: Fix this. ts_df is not defined.
            # You will need an argument to the load_ts_data function that will simply read the data
            # from the string location and return it as a dataframe as is
            ts_column = list(ts_df)[ts_column]
        if isinstance(ts_column, list):
            # If it is of type List, just pick the first one
            print("\nYou have provided a list as the 'ts_column' argument. Will pick the first value as the 'ts_column' name.")
            ts_column = ts_column[0]

        ### Now you need to save the time series column
        self.ts_column = ts_column

        # Check 'target' type
        if isinstance(target, list):
            target = target[0]
            print('    Auto_TS cannot handle Multi-Label targets. Taking first column in target list as Target = %s' %target)
        else:
            print('    Target variable given as = %s' %target)

        print("Start of loading of data.....")

        if sep is None:
            sep = ','

        ########## This is where we start the loading of the data file ######################
        if isinstance(traindata, str):
            if traindata != '':
                try:
                    ts_df = load_ts_data(traindata, self.ts_column, sep, target)
                    if isinstance(ts_df, str):
                        print("""Time Series column '%s' could not be converted to a Pandas date time column.
                            Please convert your ts_column into a pandas date-time and try again""" %self.ts_column)
                        return None
                    else:
                        if type(ts_df) == dask.dataframe.core.DataFrame:
                            print('    Dask Dataframe loaded successfully. Shape of data set = (%s,%s)' %(
                                                ts_df.shape[0].compute(),ts_df.shape[1]))
                        else:
                            print('    File loaded successfully. Shape of data set = %s' %(ts_df.shape,))
                except Exception:
                    print('File could not be loaded. Check the path or filename and try again')
                    return None
        elif isinstance(traindata, pd.DataFrame):
            print('Input is data frame. Performing Time Series Analysis')
            print(f"ts_column: {self.ts_column} sep: {sep} target: {target}")
            ts_df = load_ts_data(traindata, self.ts_column, sep, target)
            if isinstance(ts_df, str):
                print("""Time Series column '%s' could not be converted to a Pandas date time column.
                    Please convert your input into a date-time column  and try again""" %self.ts_column)
                return None
            else:
                if type(ts_df) == dask.dataframe.core.DataFrame:
                    print('    Dask Dataframe loaded successfully. Shape of data set = (%s,%s)' %(
                                        ts_df.shape[0].compute(),ts_df.shape[1]))
                else:
                    print('    pandas Dataframe loaded successfully. Shape of data set = %s' %(ts_df.shape,))
        else:
            print('File name is an empty string. Please check your input and try again')
            return None


        if ts_df.shape[1] == 1:
            ### If there is only one column, you assume that to be the target column ####
            target = list(ts_df)[0]


        preds = [x for x in list(ts_df) if x not in [self.ts_column, target]]

        if self.verbose >= 1:
            time_series_plot(ts_df[target], lags=31, title='Original Time Series',
                    chart_type='line', chart_freq=self.time_interval)
        else:
            print('No time series plot since verbose = 0. Continuing')
        ##################################################################################################
        ### Turn the time series index into a variable and calculate the difference.
        ### If the difference is not in days, then it is a hourly or minute based time series
        ### If the difference a multiple of days, then test it for weekly, monthly, quarterly, annual etc.
        ##################################################################################################
        if ts_df.index.dtype=='int' or ts_df.index.dtype=='float':
            ### You must convert the ts_df index into a date-time series using the ts_column given ####
            ts_df = ts_df.set_index(self.ts_column)
        ts_index = ts_df.index

        ## TODO: Be sure to also assign a frequency to the index column
        ## This will be helpful when finding the "future dataframe" especially for ARIMA, and ML.
        # https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases


        ##################    IF TIME INTERVAL IS NOT GIVEN DO THIS   ########################
        #### This is where the program tries to tease out the time period in the data set ####
        ######################################################################################
        if self.time_interval is None:
            print("Time Interval between observations has not been provided. Auto_TS will try to infer this now...")
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
                    print('It is a Weekly time series.')
                    self.time_interval = 'weeks'
                elif diff_in_days == 1:
                    print('It is a Daily time series.')
                    self.time_interval = 'days'
                elif 28 <= diff_in_days < 89:
                    print('It is a Monthly time series.')
                    self.time_interval = 'months'
                elif 89 <= diff_in_days < 178:
                    print('It is a Quarterly time series.')
                    self.time_interval = 'qtr'
                elif 178 <= diff_in_days < 360:
                    print('It is a Semi Annual time series.')
                    self.time_interval = 'semi'
                elif diff_in_days >= 360:
                    print('It is an Annual time series.')
                    self.time_interval = 'years'
                else:
                    print('Time Series time delta is unknown')
                    return None
            if diff_in_days == 0:
                if diff_in_hours == 0:
                    print('Time series input in Minutes or Seconds = %s' % diff_in_hours)
                    print('It is a Minute time series.')
                    self.time_interval = 'minutes'
                elif diff_in_hours >= 1:
                    print('It is an Hourly time series.')
                    self.time_interval = 'hours'
                else:
                    print('It is an Unknown Time Series delta')
                    return None
        else:
            print('Time Interval is given as %s' % self.time_interval)
            if self.time_interval in list_of_valid_time_ints:
                print('    Correct Time interval given as a valid Pandas date-range frequency...')
            else:
                print('    Error: You must give a valid time interval frequency from Pandas date-range frequency codes')
                return None

        ################# This is where you test the data and find the time interval #######
        if self.time_interval is not None:
            if self.time_interval in list_of_valid_time_ints:
                pass
            else:
                self.time_interval = self.time_interval.strip().lower()
                if self.time_interval in ['months', 'month', 'm']:
                    self.time_interval = 'M'
                elif self.time_interval in ['days', 'daily', 'd']:
                    self.time_interval = 'D'
                elif self.time_interval in ['weeks', 'weekly', 'w']:
                    self.time_interval = 'W'
                elif self.time_interval in ['qtr', 'quarter', 'q']:
                    self.time_interval = 'Q'
                elif self.time_interval in ['semi', 'semi-annual', '2q']:
                    self.time_interval = '2Q'
                elif self.time_interval in ['years', 'year', 'annual', 'y', 'a']:
                    self.time_interval = 'Y,A'
                elif self.time_interval in ['hours', 'hourly', 'h']:
                    self.time_interval = 'H'
                elif self.time_interval in ['minutes', 'minute', 'min', 'n']:
                    self.time_interval = 'M'
                elif self.time_interval in ['seconds', 'second', 'sec', 's']:
                    self.time_interval = 'S'
                else:
                    self.time_interval = 'M' # Default is Monthly
                    print('Time Interval not provided. Setting default as Monthly')
        else:
            print("(Error: 'self.time_interval' is None. This condition should not have occurred.")
            return

        # Impute seasonal_period if not provided by the user
        if self.seasonal_period is None:
            if self.time_interval == 'months':
                self.seasonal_period = 12
            elif self.time_interval == 'days':
                self.seasonal_period = 30
            elif self.time_interval in 'weeks':
                self.seasonal_period = 52
            elif self.time_interval in 'qtr':
                self.seasonal_period = 4
            elif self.time_interval in 'semi':
                self.seasonal_period = 2
            elif self.time_interval in 'years':
                self.seasonal_period = 1
            elif self.time_interval in 'hours':
                self.seasonal_period = 24
            elif self.time_interval in 'minutes':
                self.seasonal_period = 60
            elif self.time_interval in 'seconds':
                self.seasonal_period = 60
            else:
                self.seasonal_period = 12  # Default is Monthly


        ########################### This is where we store all models in a nested dictionary ##########
        mldict = lambda: defaultdict(mldict)
        self.ml_dict = mldict()
        try:
            if self.__any_contained_in_list(what_list=['best'], in_list=self.model_type):
                print(colorful.BOLD +'WARNING: Running best models will take time... Be Patient...' + colorful.END)
        except Exception:
            print('Check if your model type is a string or one of the available types of models')


        ######### This is when you need to use FB Prophet ###################################
        ### When the time interval given does not match the tested_time_interval, then use FB.
        #### Also when the number of rows in data set is very large, use FB Prophet, It is fast.
        #########                 FB Prophet              ###################################

        if self.__any_contained_in_list(what_list=['prophet', 'Prophet', 'best'], in_list=self.model_type):
            print("\n")
            print("="*50)
            print("Building Prophet Model")
            print("="*50)
            print("\n")

            name = 'Prophet'
            # Placeholder for cases when model can not be built
            score_val = np.inf
            model_build: Optional[BuildBase] = None
            model = None
            forecasts = None
            print(colorful.BOLD + '\nRunning Facebook Prophet Model...' + colorful.END)
            try:
                #### If FB prophet needs to run, it needs to be installed. Check it here ###
                model_build = BuildProphet(
                    self.forecast_period, self.time_interval, self.seasonal_period,
                    self.score_type, self.verbose, self.conf_int, self.holidays, self.growth,
                    self.seasonality)
                model, forecast_df_folds, rmse_folds, norm_rmse_folds = model_build.fit(
                    ts_df=ts_df[[target]+preds],
                    target_col=target,
                    cv = cv,
                    time_col=self.ts_column)

                # forecasts = forecast_df['yhat'].values

                ##### Make sure that RMSE works, if not set it to np.inf  #########
                if self.score_type == 'rmse':
                    score_val = rmse_folds
                else:
                    score_val = norm_rmse_folds
            except Exception as e:
                print("Exception occurred while building Prophet model...")
                print(e)
                print('    FB Prophet may not be installed or Model is not running...')

            self.ml_dict[name]['model'] = model
            self.ml_dict[name]['forecast'] = forecast_df_folds
            self.ml_dict[name][self.score_type] = score_val
            self.ml_dict[name]['model_build'] = model_build


        # if self.__any_contained_in_list(what_list=['ARIMA', 'stats', 'best'], in_list=self.model_type):
        #     ################### Let's build an ARIMA Model and add results #################
        #     print("\n")
        #     print("="*50)
        #     print("Building ARIMA Model")
        #     print("="*50)
        #     print("\n")

        #     name = 'ARIMA'

        #     # Placeholder for cases when model can not be built
        #     score_val = np.inf
        #     model_build = None
        #     model = None
        #     forecasts = None
        #     print(colorful.BOLD + '\nRunning Non Seasonal ARIMA Model...' + colorful.END)
        #     try:
        #         model_build = BuildArima(
        #             stats_scoring, p_max, d_max, q_max,
        #             forecast_period=self.forecast_period, method='mle', verbose=self.verbose
        #         )
        #         model, forecasts, rmse, norm_rmse = model_build.fit(
        #             ts_df[target]
        #         )

        #         if self.score_type == 'rmse':
        #             score_val = rmse
        #         else:
        #             score_val = norm_rmse
        #     except Exception as e:
        #         print("Exception occurred while building ARIMA model...")
        #         print(e)
        #         print('    ARIMA model error: predictions not available.')

        #     self.ml_dict[name]['model'] = model
        #     self.ml_dict[name]['forecast'] = forecasts
        #     self.ml_dict[name][self.score_type] = score_val
        #     self.ml_dict[name]['model_build'] = model_build


        if self.__any_contained_in_list(what_list=['ARIMA','arima','auto_arima','auto_SARIMAX', 'stats', 'best'], in_list=self.model_type):
            ############# Let's build a SARIMAX Model and get results ########################
            print("\n")
            print("="*50)
            print("Building Auto SARIMAX Model")
            print("="*50)
            print("\n")

            name = 'auto_SARIMAX'
            # Placeholder for cases when model can not be built
            score_val = np.inf
            model_build = None
            model = None
            forecast_df_folds = None

            print(colorful.BOLD + '\nRunning Auto SARIMAX Model...' + colorful.END)
            try:
                model_build = BuildAutoSarimax(
                    scoring=stats_scoring,
                    seasonality=self.seasonality,
                    seasonal_period=self.seasonal_period,
                    p_max=p_max, d_max=d_max, q_max=q_max,
                    forecast_period=self.forecast_period,
                    verbose=self.verbose
                )
                model, forecast_df_folds, rmse_folds, norm_rmse_folds = model_build.fit(
                    ts_df=ts_df[[target]+preds],
                    target_col=target,
                    cv = cv
                )

                if self.score_type == 'rmse':
                    score_val = rmse_folds
                else:
                    score_val = norm_rmse_folds
            except Exception as e:
                print("Exception occurred while building Auto SARIMAX model...")
                print(e)
                print('    Auto SARIMAX model error: predictions not available.')


            self.ml_dict[name]['model'] = model
            self.ml_dict[name]['forecast'] = forecast_df_folds
            self.ml_dict[name][self.score_type] = score_val
            self.ml_dict[name]['model_build'] = model_build


        if self.__any_contained_in_list(what_list=['var','Var','VAR', 'stats', 'best'], in_list=self.model_type):
            ########### Let's build a VAR Model - but first we have to shift the predictor vars ####

            if ts_df.shape[0] > 1000 and self.__any_contained_in_list(what_list=['stats', 'best'], in_list=self.model_type):
                print(colorful.BOLD + '\n===============================================' + colorful.END)
                print("Skipping VAR Model since dataset is > 1000 rows and it will take too long")
                print(colorful.BOLD + '===============================================' + colorful.END)
            else:
                print("\n")
                print("="*50)
                print("Building VAR Model - best suited for small datasets < 1000 rows and < 10 columns")
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
                        # Since we are passing ts_df there. Make sure you don't assign it
                        # back to the same variable. Make a copy and make changes to that copy.
                        ts_df_shifted = ts_df.copy(deep=True)
                        ts_df_shifted[preds] = ts_df_shifted[preds].shift(1)
                        ts_df_shifted.dropna(axis=0,inplace=True)

                        model_build = BuildVAR(scoring=stats_scoring, forecast_period=self.forecast_period, p_max=p_max, q_max=q_max)
                        model, forecasts, rmse, norm_rmse = model_build.fit(
                            ts_df_shifted[[target]+preds],
                            target_col=target,
                            cv = cv
                        )

                        if self.score_type == 'rmse':
                            score_val = rmse
                        else:
                            score_val = norm_rmse
                    except Exception as e:
                        print("Exception occurred while building VAR model...")
                        print(e)
                        warnings.warn('    VAR model error: predictions not available.')

                self.ml_dict[name]['model'] = model
                self.ml_dict[name]['forecast'] = forecasts
                self.ml_dict[name][self.score_type] = score_val
                self.ml_dict[name]['model_build'] = model_build

        if self.__any_contained_in_list(what_list=['ml', 'ML','best'], in_list=self.model_type):
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
                print(colorful.BOLD + f'\nCreating lag={self.seasonal_period} variable using target for Machine Learning model...' + colorful.END)
                ### Set the lag to be 1 since we don't need too many lagged variables for univariate case
                self.lags = self.seasonal_period
                lag = self.seasonal_period
            else:
                print(colorful.BOLD + '\nRunning Machine Learning Models...' + colorful.END)
                #### Do not create excess lagged variables for ML model ##########
                if lag <= 4:
                    lag = 4 ### set the minimum lags to be at least 4 for ML models
                elif lag >= 10:
                    lag = 10 ### set the maximum lags to be not more than 10 for ML models
                print('    Shifting %d predictors by lag=%d to align prior predictor with current target...'
                            % (len(preds), lag))
            ####### Now make sure that there is only as few lags as needed ######

            model_build = BuildML(
                scoring=self.score_type,
                forecast_period = self.forecast_period,
                ts_column = self.ts_column,
                verbose=self.verbose)
            try:
                # best = model_build.fit(ts_df=ts_df, target_col=target, lags=lag)
                model, forecasts, rmse, norm_rmse = model_build.fit(
                    ts_df=ts_df,
                    target_col=target,
                    ts_column = self.ts_column,
                    cv = cv,
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
                print("Exception occurred while building ML model...")
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
        print(colorful.BOLD + '\nBest Model is: ' + colorful.END + best_model_name)

        best_model_dict = self.ml_dict[best_model_name]
        if best_model_dict is not None:
            cv_scores = best_model_dict.get(self.score_type)
            if len(cv_scores) == 0:
                mean_cv_score =  np.inf
            else:
                mean_cv_score = self.__get_mean_cv_score(cv_scores)
        print("    Best Model (Mean CV) Score: %0.2f" % mean_cv_score) #self.ml_dict[best_model_name][self.score_type])

        end = time()
        elapsed = end-start
        print("\n\n" + "-"*50)
        print(f"Total time taken: {elapsed:.0f} seconds.")
        print("-"*50 + "\n\n")
        print("Leaderboard with best model on top of list:\n",self.get_leaderboard())
        return self

    def get_best_model_name(self) -> str:
        """
        Returns the best model name
        """
        f1_stats = {}
        for key, _ in self.ml_dict.items():
            cv_scores = self.ml_dict[key][self.score_type]

            # Standardize to a list
            if isinstance(cv_scores, np.ndarray):
                cv_scores = cv_scores.tolist()
            if not isinstance(cv_scores, List):
                cv_scores = [cv_scores]

            if len(cv_scores) == 0:
                    f1_stats[key] = np.inf
            else:
                f1_stats[key] = sum(cv_scores)/len(cv_scores)

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
        testdata,
        model: str = '',
        simple: bool = False,
        ):
        """
        Predict the results
        """

        if isinstance(model, str):
            if model == '':
                bestmodel = self.get_best_model_build()
            elif model.lower() == 'best':
                bestmodel = self.get_best_model_build()
            else:
                if self.get_model_build(model) is not None:
                    bestmodel = self.get_model_build(model)
                else:
                    print(f"(Error) Model of type '{model}' does not exist. No predictions will be made.")
                    return None
            self.model = bestmodel
        else:
            ### if no model is specified, just use the best model ###
            bestmodel = self.get_best_model_build()
            self.model = bestmodel


        if isinstance(testdata, pd.Series) or isinstance(testdata, pd.DataFrame):

            # During training, we internally converted a column datetime index to the dataframe date time index
            # We need to do the same while predicing for consistence
            if (model == 'ML') or self.get_best_model_name() == 'ML' or (model == 'best' and self.get_best_model_name() == 'ML'):
                if self.ts_column in testdata.columns:
                    testdata.set_index(self.ts_column, inplace=True)
                elif self.ts_column in testdata.index.name:
                    pass
                else:
                    print(f"(Error) Model to be used for prediction 'ML'. Hence, X_egogen' must have a column (or index) called '{self.ts_column}' corresponding to the original ts_index column passed during training. No predictions will be made.")
                    return None
                ### Now do the predictions using the final model asked to be predicted ###
                predictions = bestmodel.predict(testdata,simple=simple)
            elif model.lower() == 'prophet' or self.get_best_model_name() == 'Prophet':
                predictions = bestmodel.predict(testdata,simple=simple)
            elif self.get_best_model_name() == 'VAR':
                predictions = bestmodel.predict(testdata,simple=simple)
            else:
                predictions = bestmodel.predict(testdata,simple=simple)
        elif isinstance(testdata, int):
            #### if testdata is an Integer, then it appears to be a forecast period, then use it that way
            ### only certain stats-based models can use forecast period
            if (model == 'ML') or (model == 'best' and self.get_best_model_name() == 'ML'):
                print(f'{model} is an ML-based model, hence it cannot be used with a forecast period')
                predictions = None
            else:
                predictions = bestmodel.predict(testdata,simple=simple)
        else:
            ### if there is no testdata, at least they must give forecast_period
            if testdata is None:
                print('If test_data is None, then forecast_period must be given')
                return
        return predictions

    def get_leaderboard(self, ascending=True) -> Optional[pd.DataFrame]:
        """
        Returns the leaderboard after fitting
        """
        names = []
        mean_cv_scores = []
        # std_cv_scores = [] # TODO: Add later
        local_ml_dict = self.ml_dict
        if local_ml_dict is not None:
            model_names = list(local_ml_dict.keys())
            if model_names != []:
                for model_name in model_names:
                    names.append(model_name)
                    model_dict_single_model = local_ml_dict.get(model_name)
                    if model_dict_single_model is not None:
                        cv_scores = model_dict_single_model.get(self.score_type)
                        #### This is quite complicated since we have to make sure it doesn't blow up
                        if cv_scores == np.inf:
                            mean_cv_score =  np.inf
                        elif isinstance(cv_scores, list):
                            if len(cv_scores) == 0:
                                mean_cv_score =  np.inf
                        else:
                            mean_cv_score = self.__get_mean_cv_score(cv_scores)
                        # if isinstance(cv_scores, float):
                        #     mean_cv_score = cv_scores
                        # else: # Assuming List
                        #     mean_cv_score = sum(cv_scores)/len(cv_scores)
                        mean_cv_scores.append(mean_cv_score)

                results = pd.DataFrame({"name": names, self.score_type: mean_cv_scores})
                results.sort_values(self.score_type, ascending=ascending, inplace=True)
                return results
            else:
                return None
        else:
            return None

    def plot_cv_scores(self, **kwargs):
        """
        Plots a boxplot of the cross validation scores for the various models.
        **kwargs: Keyword arguments to be passed to the seaborn boxplot call
        """
        cv_df = self.get_cv_scores()
        ax = sns.boxplot(x="Model", y="CV Scores", data=cv_df, **kwargs)
        return ax

    def get_cv_scores(self) -> pd.DataFrame:
        """
        Return a tidy data frame with the CV scores across all models
        :rtype pandas.DataFrame
        """
        cv_df = pd.DataFrame(columns=['Model', 'CV Scores'])
        for model in self.ml_dict.keys():
            rmses = self.ml_dict.get(model).get("rmse")
            new_row = {'Model':model, 'CV Scores':rmses}
            cv_df = cv_df.append(new_row, ignore_index=True)
            # print(f"Model: {model} RMSEs: {rmses}")
        cv_df = cv_df.explode('CV Scores').reset_index(drop=True)
        cv_df = cv_df.astype({"CV Scores": float})
        return cv_df

    def __get_mean_cv_score(self, cv_scores: Union[float, List]):
        """
        If gives a list fo cv scores, this will return the mean cv score
        If cv_score is a float (single value), it simply returns that
        """
        if isinstance(cv_scores, float):
            mean_cv_score = cv_scores
        else: # Assuming List
            mean_cv_score = sum(cv_scores)/len(cv_scores)
        return mean_cv_score

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
#################################################################################
module_type = 'Running' if  __name__ == "__main__" else 'Imported'
version_number = '0.0.35'
print(f"""{module_type} auto_timeseries version:{version_number}. Call by using:
model = auto_timeseries(score_type='rmse',
                time_interval='M',
                non_seasonal_pdq=None, seasonality=False, seasonal_period=12,
                model_type=['best'],
                verbose=2)
model.fit(traindata, ts_column,target)
model.predict(testdata, model='best')
""")
#################################################################################
