##########################################################
#Defining AUTO_TIMESERIES here
##########################################################
module_type = 'Running' if  __name__ == "__main__" else 'Imported'
version_number = '0.0.24'
print(f"Running Auto Timeseries version: {version_number}")

# Call by using:
#     import auto_ts as AT
#     automl_model = AT.AutoTimeSeries(
#         score_type='rmse',
#         forecast_period=4,
#         time_interval='Week',
#         non_seasonal_pdq=None,
#         seasonality=True,
#         seasonal_period=52,
#         model_type=['SARIMAX','ML'],
#         verbose=0)
# automl_model.fit(train, 'WeekDate', '9L Cases', ',')
# automl_model.get_leaderboard()

# To run all models (Stats, ML, FB Prophet, etc.) set model_type='best'""")

# print("To remove previous versions, perform 'pip uninstall auto_ts'")
# print('To get the latest version, perform "pip install auto_ts --no-cache-dir --ignore-installed"')




import warnings
from typing import List, Dict, Optional, Tuple, Union

from datetime import datetime
import copy
import pdb
from collections import defaultdict
import operator
from time import time

# Tabular Data
import pandas as pd  # type: ignore
import numpy as np  # type: ignore

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
                   time_series_plot, print_static_rmse, print_dynamic_rmse


class AutoTimeSeries:
    def __init__(
        self,
        forecast_period: int,
        score_type: str = 'rmse',
        time_interval: Optional[str] = None,
        non_seasonal_pdq: Optional[Tuple]=None,
        seasonality: bool = False,
        seasonal_period: int = 12,
        conf_int: float = 0.95,
        model_type: Union[str, List] = "stats",
        verbose: int = 0
    ):
        """
        Initialize an AutoTimeSeries object

        :forecast_period The number of time intervals ahead that you want to forecast
        :type forecast_period int

        :score_type: The metric used for scoring the models. Default = 'rmse'
        Currently only 2 are supported:
        (1) RMSE
        (2) Normalized RMSE (ratio of RMSE to the standard deviation of actuals)
        :type str

        :param time_interval Used to indicate the frequency at which the data is collected
        This is used for 2 purposes (1) in building the Prophet model and (2) used to impute
        the seasonal period for SARIMAX in case it is not provided by the user (None)
        Allowed values are:
          (1) 'months', 'month', 'm' for monthly frequency data
          (2) 'days', 'daily', 'd' for daily freuency data
          (3) 'weeks', 'weekly', 'w' for weekly frequency data
          (4) 'qtr', 'quarter', 'q' for quarterly frequency data
          (5) 'years', 'year', 'annual', 'y', 'a' for yearly frequency data
          (6) 'hours', 'hourly', 'h' for hourly frequency data
          (7) 'minutes', 'minute', 'min', 'n' for minute frequency data
          (8) 'seconds', 'second', 'sec', 's' for second frequency data
        :type time_interval Optional[str]

        :param: non_seasonal_pdq Indicates the maximum value of p, d, q to be used in the search for "stats" based models.
        If None, then the following values are assumed max_p = 3, max_d = 1, max_q = 3
        :type non_seasonal_pdq Optional[Tuple]

        :param seasonality Used in the building of the SARIMAX model only at this time.
        :type seasonality bool

        :param seasonal_period: Indicates the seasonality period in the data.
        Used in the building of the SARIMAX model only at this time.
        There is no impact of this argument if seasonality is set to False
        If None, the program will try to inder this from the time_interval (frequency) of the data
        (1) If frequency = Monthly, then seasonal_period = 12
        (1) If frequency = Daily, then seasonal_period = 30
        (1) If frequency = Weekly, then seasonal_period = 52
        (1) If frequency = Quarterly, then seasonal_period = 4
        (1) If frequency = Yearly, then seasonal_period = 1
        (1) If frequency = Hourly, then seasonal_period = 24
        (1) If frequency = Minutes, then seasonal_period = 60
        (1) If frequency = Seconds, then seasonal_period = 60
        :type seasonal_period int

        :param conf_int: Confidence Interval for building the Prophet model. Default: 0.95
        :type conf_int float

        :param model_type The type(s) of model to build. Default to building only statistical models
        Can be a string or a list of models. Allowed values are:
        'best', 'prophet', 'pyflux', 'stats', 'ARIMA', 'SARIMAX', 'VAR', 'ML'.
        "prophet" will build a model using FB Prophet -> this means you must have FB Prophet installed
        "stats" will build statsmodels based ARIMA, SARIMAX and VAR models
        "ML" will build a machine learning model using Random Forests provided explanatory vars are given
        'best' will try to build all models and pick the best one
        If a list is provided, then only those models will be built
        WARNING: "best" might take some time for large data sets. We recommend that you
        choose a small sample from your data set bedfore attempting to run entire data.
        :type model_type: Union[str, List]

        :param verbose Indicates the verbosity of printing (Default = 0)
        :type verbose int

        ####################################################################################
        ####                          Auto Time Series                                  ####
        ####                    Conceived and Developed by Ram Seshadri                 ####
        ####                        All Rights Reserved                                 ####
        ####################################################################################
        ##################################################################################################
        AUTO_TIMESERIES IS A COMPLEX MODEL BUILDING UTILITY FOR TIME SERIES DATA. SINCE IT AUTOMATES MANY
        TASKS INVOLVED IN A COMPLEX ENDEAVOR, IT ASSUMES MANY INTELLIGENT DEFAULTS. BUT YOU CAN CHANGE THEM.
        Auto_Timeseries will rapidly build predictive models based on Statsmodels ARIMA, Seasonal ARIMA
        and Scikit-Learn ML. It will automatically select the BEST model which gives best score specified.
        #####################################################################################################
        """

        self.ml_dict: Dict = {}
        self.score_type: str = score_type
        self.forecast_period = forecast_period
        self.time_interval = time_interval
        self.non_seasonal_pdq = non_seasonal_pdq
        self.seasonality = seasonality
        self.seasonal_period = seasonal_period
        self.conf_int = conf_int
        if isinstance(model_type, str):
            model_type = [model_type]
        self.model_type = model_type
        self.verbose = verbose

        self.allowed_models = ['best', 'prophet', 'pyflux', 'stats', 'ARIMA', 'SARIMAX', 'auto_SARIMAX', 'VAR', 'ML']

    def fit(
        self,
        traindata: Union[str, pd.DataFrame],
        ts_column: Union[str, int, List[str]],
        target: Union[str, List[str]],
        cv: Optional[int]=None,
        sep: Optional[str]=None):
        """
        Train the AutoTimeseries object
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
            If None, a single fold is used
        :type cv Optional[int]

        :param sep: Note that optionally you can give a separator for the data in your file.
            Default is None which treats the sep as a comma (datafile as a 'csv').
        :type sep Optional[str]
        """

        start = time()
        print("Start of Fit.....")

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
            ts_column = list(ts_df)[ts_column]
        if isinstance(ts_column, list):
            # If it is of type List, just pick the first one
            print("\nYou have provided a list as the 'ts_column' argument. Will pick the first value as the 'ts_column' name.")
            ts_column = ts_column[0]


        self.ts_column = ts_column

        # Check 'target' type
        if isinstance(target, list):
            target = target[0]
            print('    Taking the first column in target list as Target variable = %s' %target)
        else:
            print('    Target variable = %s' %target)

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
                            Please convert your input into a date-time column  and try again""" %self.ts_column)
                        return None
                    else:
                        print('    File loaded successfully. Shape of data set = %s' %(ts_df.shape,))
                except:
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
                print('    Dataframe loaded successfully. Shape of data set = %s' %(ts_df.shape,))
        else:
            print('File name is an empty string. Please check your input and try again')
            return None


        if ts_df.shape[1] == 1:
            ### If there is only one column, you assume that to be the target column ####
            target = list(ts_df)[0]


        preds = [x for x in list(ts_df) if x not in [self.ts_column, target]]

        ##################################################################################################
        ### Turn the time series index into a variable and calculate the difference.
        ### If the difference is not in days, then it is a hourly or minute based time series
        ### If the difference a multiple of days, then test it for weekly, monthly, qtrly, annual etc.
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
            print("Time Interval of obserations has not been provided. Program will try to figure this out now...")
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
                    self.time_interval = 'qtr'
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

        ################# This is where you test the data and find the time interval #######
        if self.time_interval is not None:
            self.time_interval = self.time_interval.strip().lower()
            if self.time_interval in ['months', 'month', 'm']:
                self.time_interval = 'months'
            elif self.time_interval in ['days', 'daily', 'd']:
                self.time_interval = 'days'
            elif self.time_interval in ['weeks', 'weekly', 'w']:
                self.time_interval = 'weeks'
            elif self.time_interval in ['qtr', 'quarter', 'q']:
                self.time_interval = 'qtr'
            elif self.time_interval in ['years', 'year', 'annual', 'y', 'a']:
                self.time_interval = 'years'
            elif self.time_interval in ['hours', 'hourly', 'h']:
                self.time_interval = 'hours'
            elif self.time_interval in ['minutes', 'minute', 'min', 'n']:
                self.time_interval = 'minutes'
            elif self.time_interval in ['seconds', 'second', 'sec', 's']:
                self.time_interval = 'seconds'
            else:
                self.time_interval = 'months' # Default is Monthly
        else:
            print("(Error: 'self.time_interval' is None. This condition should not have occured.")
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
        except:
            print('Check if your model type is a string or one of the available types of models')


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
                    self.forecast_period, self.time_interval,
                    self.score_type, self.verbose, self.conf_int)
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
                print("Exception occured while building Prophet model...")
                print(e)
                print('    FB Prophet may not be installed or Model is not running...')

            self.ml_dict[name]['model'] = model
            self.ml_dict[name]['forecast'] = forecast_df_folds
            self.ml_dict[name][self.score_type] = score_val
            self.ml_dict[name]['model_build'] = model_build


        if self.__any_contained_in_list(what_list=['pyflux','ARIMA', 'stats', 'best'], in_list=self.model_type):
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
                model, forecasts, rmse, norm_rmse = model_build.fit(
                    ts_df[target]
                )

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


        if self.__any_contained_in_list(what_list=['SARIMAX','auto_SARIMAX', 'stats', 'best'], in_list=self.model_type):
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
                print("Exception occured while building Auto SARIMAX model...")
                print(e)
                print('    Auto SARIMAX model error: predictions not available.')


            self.ml_dict[name]['model'] = model
            self.ml_dict[name]['forecast'] = forecast_df_folds
            self.ml_dict[name][self.score_type] = score_val
            self.ml_dict[name]['model_build'] = model_build
















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
        print(colorful.BOLD + '\nBest Model is: ' + colorful.END + best_model_name)

        loBestModelDict = self.ml_dict[best_model_name]
        if loBestModelDict is not None:
            cv_scores = loBestModelDict.get(self.score_type)
            mean_cv_score = self.__get_mean_cv_score(cv_scores)
        print("    Best Model (Mean CV) Score: %0.2f" % mean_cv_score) #self.ml_dict[best_model_name][self.score_type])
        # print("    Best Model Forecasts (Validation Set):")
        # print(self.ml_dict[best_model_name]['forecast'])

        end = time()

        print("\n\n" + "-"*50)
        print(f"Total time taken: {end-start} seconds.")
        print("-"*50 + "\n\n")

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
        model: str = 'best',
        X_exogen: Optional[pd.DataFrame]=None,
        forecast_period: Optional[int] = None,
        simple: bool = True) -> Optional[np.array]:
        """
        Predict the results
        """

        if X_exogen is not None:
            # During training, we internally converted a column datetime index to the dataframe date time index
            # We need to do the same while predicing for consistence

            if (model == 'ML') or (model == 'best' and self.get_best_model_name() == 'ML'):
                if self.ts_column in X_exogen.columns:
                    X_exogen.set_index(self.ts_column, inplace=True)
                elif self.ts_column in X_exogen.index.name:
                    pass
                else:
                    print(f"(Error) Model to be used for prediction 'ML'. Hence, X_egogen' must have a column (or index) called '{self.ts_column}' corresponding to the original ts_index column passed during training. No predictions will be made.")
                    return None

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
            print(f"(Error) Model of type '{model}' does not exist. No predictions will be made.")
            predictions = None

        return predictions

    def get_leaderboard(self, ascending=True) -> Optional[pd.DataFrame]:
        """
        Returns the leaderboard after fitting
        """
        names = []
        mean_cv_scores = []
        # std_cv_scores = [] # TODO: Add later
        loMlDict = self.ml_dict
        if loMlDict is not None:
            model_names = list(loMlDict.keys())
            if model_names is not None:
                for model_name in model_names:
                    names.append(model_name)
                    loModelDictSingleModel = loMlDict.get(model_name)
                    if loModelDictSingleModel is not None:
                        cv_scores = loModelDictSingleModel.get(self.score_type)
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
        cv_df = df = pd.DataFrame(columns=['Model', 'CV Scores'])
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
