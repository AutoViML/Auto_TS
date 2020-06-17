import warnings
from typing import Dict, Optional

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
# from .models import build_arima_model, build_sarimax_model, build_var_model, \
#                     build_pyflux_model, build_prophet_model, run_ensemble_model
from .models import build_arima_model, build_pyflux_model
from .models import BuildSarimax, BuildVAR, BuildML
from .models.build_prophet import BuildProphet


# Utils
from .utils import colorful, load_ts_data, convert_timeseries_dataframe_to_supervised, \
                   time_series_plot, print_static_rmse, print_dynamic_rmse
#from .utils import colors, eda, etl, metrics, val


class AutoTimeseries:
    def __init__(self, score_type: str ='rmse',
                forecast_period: int = 5, time_interval: str = '', non_seasonal_pdq=None,
                seasonality: bool = False, seasonal_period: int = 12, seasonal_PDQ=None,
                conf_int: float = 0.95, model_type: str ="stats", verbose: int =0):
        """
        Initializae an AutoTimeSeries object
        # TODO: Add complete docstring
        # TODO: Add object types

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
        self.seasonal_PDQ = seasonal_PDQ
        self.conf_int = conf_int
        self.model_type = model_type
        self.verbose = verbose

    def fit(self, traindata, ts_column, target, sep=','):
        """
        Train the AutoTimeseries object
        # TODO: Complete docstring
        """
            
        # start_time = time.time()  # Unused

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
        ################################
        # TODO: #8 Check: seasonal_order is not used anywhere in the code, hence commented for now.
        # if type(self.seasonal_PDQ) == tuple:
        #     seasonal_order = copy.deepcopy(self.seasonal_PDQ)
        # else:
        #     seasonal_order = (3, 1, 3)

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
                    return
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
                    return
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
            if self.model_type.lower() == 'best':
                print(colorful.BOLD +'WARNING: Running best models will take time... Be Patient...' + colorful.END)
        except:
            print('Check if your model type is a string or one of the available types of models')
        ######### This is when you need to use FB Prophet ###################################
        ### When the time interval given does not match the tested_time_interval, then use FB.
        #### Also when the number of rows in data set is very large, use FB Prophet, It is fast.
        #########                 FB Prophet              ###################################

        if self.model_type.lower() in ['prophet','best']:
            print("\n")
            print("="*50)
            print("Building Prophet Model")
            print("="*50)
            print("\n")

            name = 'FB_Prophet'
            print(colorful.BOLD + '\nRunning Facebook Prophet Model...' + colorful.END)
            # try:
            #### If FB prophet needs to run, it needs to be installed. Check it here ###
            # model, forecast_df, rmse, norm_rmse = build_prophet_model(
            #                             ts_df, ts_column, target, self.forecast_period, self.time_interval,
            #                             self.score_type, self.verbose, self.conf_int)
            prophet_model = BuildProphet(self.forecast_period, self.time_interval,
                                         self.score_type, self.verbose, self.conf_int)
            model, forecast_df, rmse, norm_rmse = prophet_model.fit(
                                         ts_df, ts_column, target)

            self.ml_dict[name]['model'] = model
            self.ml_dict[name]['forecast'] = forecast_df['yhat'].values
            ##### Make sure that RMSE works, if not set it to np.inf  #########
            if self.score_type == 'rmse':
                score_val = rmse
            else:
                score_val = norm_rmse
            # except:
            #     print('    FB Prophet may not be installed or Model is not running...')
            #     score_val = np.inf
            self.ml_dict[name][self.score_type] = score_val
            self.ml_dict[name]['model_build'] = prophet_model
        
        if self.model_type.lower() in ['stats','best']:
            print("\n")
            print("="*50)
            print("Building PyFlux Model")
            print("="*50)
            print("\n")

            ##### First let's try the following models in sequence #########################################
            nsims = 100   ### this is needed only for M-H models in PyFlux
            name = 'PyFlux'
            print(colorful.BOLD + '\nRunning PyFlux Model...' + colorful.END)
            try:
                self.ml_dict[name]['model'], self.ml_dict[name]['forecast'], rmse, norm_rmse = \
                    build_pyflux_model(ts_df, target, p_max, q_max, d_max, self.forecast_period,
                                    'MLE', nsims, self.score_type, self.verbose)
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
            if self.score_type == 'rmse':
                score_val = rmse
            else:
                score_val = norm_rmse
            self.ml_dict[name][self.score_type] = score_val
            self.ml_dict[name]['model_build'] = None  # TODO: Add the right value here
            ################### Let's build an ARIMA Model and add results #################
            
            print("\n")
            print("="*50)
            print("Building ARIMA Model")
            print("="*50)
            print("\n")

            name = 'ARIMA'
            print(colorful.BOLD + '\nRunning Non Seasonal ARIMA Model...' + colorful.END)
            try:
                self.ml_dict[name]['model'], self.ml_dict[name]['forecast'], rmse, norm_rmse = build_arima_model(ts_df[target],
                                                        stats_scoring,p_max,d_max,q_max,
                                        forecast_period=self.forecast_period,method='mle',verbose=self.verbose)
            except:
                print('    ARIMA model error: predictions not available.')
                score_val = np.inf
            if self.score_type == 'rmse':
                score_val = rmse
            else:
                score_val = norm_rmse
            self.ml_dict[name][self.score_type] = score_val
            self.ml_dict[name]['model_build'] = None  # TODO: Add the right value here
            ############# Let's build a SARIMAX Model and get results ########################
            
            print("\n")
            print("="*50)
            print("Building SARIMAX Model")
            print("="*50)
            print("\n")

            name = 'SARIMAX'
            print(colorful.BOLD + '\nRunning Seasonal SARIMAX Model...' + colorful.END)
            # try:
            # self.ml_dict[name]['model'], self.ml_dict[name]['forecast'], rmse, norm_rmse = build_sarimax_model(
            #     ts_df[target], stats_scoring, self.seasonality,
            #     self.seasonal_period, p_max, d_max, q_max,
            #     self.forecast_period,self.verbose
            # )
            sarimax_model = BuildSarimax(
                metric=stats_scoring,
                seasonality=self.seasonality,
                seasonal_period=self.seasonal_period,
                p_max=p_max, d_max=d_max, q_max=q_max,
                forecast_period=self.forecast_period,
                verbose=self.verbose
            )
            # TODO: https://github.com/AutoViML/Auto_TS/issues/10
            self.ml_dict[name]['model'], self.ml_dict[name]['forecast'], rmse, norm_rmse = sarimax_model.fit(ts_df[target])

            # except:
            #     print('    SARIMAX model error: predictions not available.')
            #     score_val = np.inf
            if self.score_type == 'rmse':
                score_val = rmse
            else:
                score_val = norm_rmse
            self.ml_dict[name][self.score_type] = score_val
            self.ml_dict[name]['model_build'] = sarimax_model

            ########### Let's build a VAR Model - but first we have to shift the predictor vars ####

            print("\n")
            print("="*50)
            print("Building VAR Model")
            print("="*50)
            print("\n")

            name = 'VAR'
            var_model = None # Placeholder for cases when model can not be built
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

                        var_model = BuildVAR(criteria=stats_scoring, forecast_period=self.forecast_period, p_max=p_max, q_max=q_max)
                        self.ml_dict[name]['model'], self.ml_dict[name]['forecast'], rmse, norm_rmse = var_model.fit(ts_df[[target]+preds])
                    else:
                        print(colorful.BOLD + '\nNo predictors available. Skipping VAR model...' + colorful.END)
                        score_val = np.inf
                except:
                    warnings.warn('    VAR model error: predictions not available.')
                    rmse = np.inf
                    norm_rmse = np.inf
            ################################################################
            if self.score_type == 'rmse':
                score_val = rmse
            else:
                score_val = norm_rmse
            ########################################################################
            self.ml_dict[name][self.score_type] = score_val
            self.ml_dict[name]['model_build'] = var_model  # TODO: Add the right value here
        
        if self.model_type.lower() in ['ml','best']:
            ########## Let's build a Machine Learning Model now with Time Series Data ################
            
            print("\n")
            print("="*50)
            print("Building ML Model")
            print("="*50)
            print("\n")
            
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
                        train = dfxs[:-self.forecast_period]
                        test = dfxs[-self.forecast_period:]

                        ml_model = BuildML()
                        best = ml_model.fit(train[preds], train[target], 'TimeSeries', self.score_type, self.verbose)
                        bestmodel = best[0]
                        self.ml_dict[name]['model'] = bestmodel
                        ### Certain models dont have random state => so dont do this for all since it will error
                        #best.set_params(random_state=0)
                        self.ml_dict[name]['forecast'] = bestmodel.fit(train[preds],train[target]).predict(test[preds])
                        rmse, norm_rmse = print_dynamic_rmse(test[target].values,
                                                    bestmodel.predict(test[preds]),
                                                    train[target].values)
                        #### Plotting actual vs predicted for RF Model #################
                        plt.figure(figsize=(5, 5))
                        plt.scatter(train.append(test)[target].values,
                                    np.r_[bestmodel.predict(train[preds]), bestmodel.predict(test[preds])])
                        plt.xlabel('Actual')
                        plt.ylabel('Predicted')
                        plt.show(block=False)
                        ############ Draw a plot of the Time Series data ######
                        time_series_plot(dfxs[target], chart_time=self.time_interval)
                    else:
                        print(colorful.BOLD + '\nNo predictors available. Skipping Machine Learning model...' + colorful.END)
                        score_val = np.inf
                except:
                    print('    For ML model, evaluation score is not available.')
                    score_val = np.inf
            ################################################################
            if self.score_type == 'rmse':
                score_val = rmse
            else:
                score_val = norm_rmse
                rmse = np.inf
                norm_rmse = np.inf
            ########################################################################
            self.ml_dict[name][self.score_type] = score_val
            self.ml_dict[name]['model_build'] = None  # TODO: Add the right value here
            
        if not self.model_type.lower() in ['stats','ml', 'prophet', 'best']:
            print('The model_type should be either stats, prophet, ml or best. Check your input and try again...')
            return self.ml_dict
        ######## Selecting the best model based on the lowest rmse score ######
        # f1_stats = {}
        # for key, _ in self.ml_dict.items():
        #     f1_stats[key] = self.ml_dict[key][self.score_type]
        best_model_name = self.get_best_model_name()    # min(f1_stats.items(), key=operator.itemgetter(1))[0]
        print(colorful.BOLD + '\nBest Model is:' + colorful.END)
        print('    %s' % best_model_name)
        # best_model = self.ml_dict[best_model_name]['model']  # unused
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

    def predict(self, model: str = 'best') -> Optional[np.array]:
        """
        Predict the results
        """
        print("This function has not been implemented yet. But the idea would be that this would make the prediction using the best model or the model type passed as an argument.")
        return None

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
