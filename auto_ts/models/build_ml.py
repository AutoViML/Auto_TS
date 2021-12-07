import warnings
import copy
from typing import List, Optional, Tuple

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from pandas.core.generic import NDFrame # type:ignore
import matplotlib.pyplot as plt

#from tscv import GapWalkForward # type: ignore
from sklearn.model_selection import TimeSeriesSplit

# imported ML models from scikit-learn
from sklearn.model_selection import (ShuffleSplit, StratifiedShuffleSplit, # type: ignore
                                    TimeSeriesSplit, cross_val_score) # type: ignore
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis # type: ignore
from sklearn.ensemble import (BaggingRegressor, ExtraTreesRegressor,  # type: ignore
                             BaggingClassifier, ExtraTreesClassifier,  # type: ignore
                             AdaBoostRegressor, AdaBoostClassifier, # type: ignore
                             RandomForestClassifier, RandomForestRegressor) # type: ignore

from xgboost import XGBRegressor, XGBClassifier
from xgboost import plot_importance
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeCV # type: ignore
from sklearn.svm import LinearSVC, SVR, LinearSVR # type: ignore
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier  # type: ignore

import dask
import dask.dataframe as dd
import dask_xgboost
import xgboost as xgb
from dask.distributed import Client, progress
import psutil


################################################################################################
from .build_base import BuildBase

# helper functions
from ..utils import print_static_rmse, print_dynamic_rmse, convert_timeseries_dataframe_to_supervised, print_ts_model_stats
from ..utils.etl import change_to_datetime_index, change_to_datetime_index_test, reduce_mem_usage, load_test_data
#################################################################################################
import pdb
import time

class BuildML(BuildBase):
    def __init__(self, scoring: str = '', forecast_period: int = 2, ts_column: str = '', 
                        time_interval: str = '', sep: str = ',', dask_xgboost_flag: int = 0, verbose: int = 0):
        """
        Automatically build a ML Model
        """
        super().__init__(
            scoring=scoring,
            forecast_period=forecast_period,
            verbose=verbose
        )

        # Specific to ML model
        # These are needed so that during prediction later, the data can be transformed correctly
        self.lags: int = 0
        self.univariate = False
        self.ts_column = ts_column
        self.time_interval = time_interval
        self.sep = sep
        self.dask_xgboost_flag = dask_xgboost_flag    

        self.transformed_target: str = ""
        self.transformed_preds: List[str] = []

        # This saves the last `self.lags` of the original train dataframe
        # This is needed during predictions to transform the X_test
        # to a supervised learning problem.
        self.df_train_prepend = pd.DataFrame()
        self.train_df = pd.DataFrame()


    def fit(self, ts_df: pd.DataFrame, target_col: str, ts_column:str, cv: Optional[int]=None, lags: int = 0):
        """
        Build a Time Series Model using Machine Learning models.
        Quickly builds and runs multiple models for a clean data set (only numerics).
        """
        ts_df = copy.deepcopy(ts_df)
        self.original_target_col = target_col
        self.lags = lags
        self.original_preds = [x for x in list(ts_df) if x not in [self.original_target_col]]
        self.ts_column = ts_column
        
        if len(self.original_preds) > 0:
            self.univariate = False
            if type(ts_df) == dask.dataframe.core.DataFrame:
                continuous_vars = ts_df.select_dtypes('number').columns.tolist()
                numvars = [x for x in continuous_vars if x not in [target_col]]
                preds = [x for x in list(ts_df) if x not in [self.original_target_col]]
                catvars = ts_df.select_dtypes('object').columns.tolist() + ts_df.select_dtypes('category').columns.tolist()
                if len(catvars) > 0:
                    print('    Warning: Dropping Categorical variables %s. You can Label Encode them and try ML modeling again...' %catvars)
            else:
                features_dict = classify_features(ts_df, self.original_target_col)
                cols_to_remove = features_dict['cols_delete'] + features_dict['IDcols'] + features_dict['discrete_string_vars']
                preds = [x for x in list(ts_df) if x not in [self.original_target_col]+cols_to_remove]
                catvars = ts_df[preds].select_dtypes(include = 'object').columns.tolist() + ts_df[preds].select_dtypes(include = 'category').columns.tolist()
                numvars = ts_df[preds].select_dtypes(include = 'number').columns.tolist()
            if len(catvars) > 0:
                print('    Warning: Dropping Categorical variables %s. You can Label Encode them and try ML modeling again...' %catvars)
            self.original_preds = numvars
            preds = copy.deepcopy(numvars)
            if len(numvars) > 30:
                print('    Warning: Too many continuous variables. Hence numerous lag features will be generated. ML modeling may take time...')
        else:
            self.univariate = True
            preds = self.original_preds[:]

        ts_df = ts_df[preds+[self.original_target_col]]

        # Order data
        ts_df = self.order_df(ts_df)

        if type(ts_df) == dask.dataframe.core.DataFrame:
            num_obs = ts_df.shape[0].compute()
        else:
            num_obs = ts_df.shape[0]

        #self.train_df = ts_df  # you don't want to store the entire train_df in model
        # Convert to supervised learning problem
        dfxs, self.transformed_target, self.transformed_preds = self.df_to_supervised(
                ts_df=ts_df, drop_zero_var = True)


        print("\nFitting ML model")
        print('    %d variables used in training ML model = %s' %(
                                len(self.transformed_preds),self.transformed_preds))

        #if len(self.transformed_preds) > 1:
        #    self.univariate = False
        #else:
        #    self.univariate = True
        # print(f"Transformed DataFrame:")
        # print(dfxs.info())
        # print(f"Transformed Target: {self.transformed_target}")
        # print(f"Transformed Predictors: {self.transformed_preds}")


        #######################################
        #### Cross Validation across Folds ####
        #######################################

        rmse_folds = []
        norm_rmse_folds = []
        forecast_df_folds = []  # TODO: See if this can be retreived somehow
        ### Creating a new way to skip cross validation when trying to run auto-ts multiple times. ###
        if cv == 0:
            cv_in = 0
        else:
            cv_in = copy.deepcopy(cv)
        NFOLDS = self.get_num_folds_from_cv(cv)
        ###########################################################################
        if self.forecast_period <= 5:
            #### Set a minimum of 5 for the number of rows in test!
            self.forecast_period = 5
        ### In case the number of forecast_period is too high, just reduce it so it can fit into num_obs
        if NFOLDS*self.forecast_period > num_obs:
            self.forecast_period = int(num_obs/10)
            print('    Forecast period too high. Cutting by 90%% to %d to enable cross_validation' %self.forecast_period)
        ###########################################################################

        seed = 99

        X_train = dfxs[self.transformed_preds]
        y_train = dfxs[self.transformed_target]
        dft = dfxs[self.transformed_preds+[self.transformed_target]]

        # Decide NUM_ESTIMATORS for trees
        if len(X_train) <= 100000 or dft.shape[1] < 50:
            NUMS = 200
        else:
            NUMS = 100

        if self.scoring == '':
            self.scoring = 'neg_root_mean_squared_error'
        elif self.scoring == 'rmse':
            self.scoring = 'neg_root_mean_squared_error'

        print('\nRunning Cross Validation using XGBoost model..')
        ###########################################################################################
        #cv = GapWalkForward(n_splits=NFOLDS, gap_size=0, test_size=self.forecast_period)
        max_trainsize = len(dft) - self.forecast_period
        try:
            cv = TimeSeriesSplit(n_splits=NFOLDS, test_size=self.forecast_period) ### this works only sklearn v 0.0.24]
        except:
            cv = TimeSeriesSplit(n_splits=NFOLDS, max_train_size = max_trainsize)

        print('    Max. iterations using expanding window cross validation = %d' %NFOLDS)
        start_time = time.time()
        rmse_folds = []
        norm_rmse_folds = []
        y_trues = copy.deepcopy(y_train)
        concatenated = pd.DataFrame()
        extra_concatenated = pd.DataFrame()
        bst_models = []
        important_features = []
        if len(y_trues) <= 1000:
            num_boost_round =300
        elif len(y_trues) > 1000 and len(y_trues) <= 10000:
            num_boost_round = 400
        elif len(y_trues) > 10000 and len(y_trues) <= 100000:
            num_boost_round = 500
        else:
            num_boost_round = 700
        #########################################################################################################
        ################    M  O  D  E  L    H Y P E R   P A R A M    T U N I N G    T A K E S   P L A C E   ####
        ######################################################################################################### 
        if type(dft) == dask.dataframe.core.DataFrame:
            #### We use the gridsearchCV from dask_ml here  ###
            from dask_ml.model_selection import GridSearchCV
            if  cv_in == 0:
                print('Skipping cross validation steps since cross_validation = %s' %cv_in)
            else:
                #dft = dft.head(len(dft)) ### this converts dask into a pandas dataframe
                ### check available memory and allocate at least 1GB of it in the Client in DASK #############################
                memory_free = str(max(1, int(psutil.virtual_memory()[0]/1e9)))+'GB'
                print('    Using Dask XGBoost algorithm with %s virtual CPUs and %s memory limit...' %(get_cpu_worker_count(), memory_free))
                client = Client(n_workers=get_cpu_worker_count(), threads_per_worker=1, processes=True, silence_logs=50,
                                memory_limit=memory_free)
                print('    Dask client configuration: %s' %client)
                print('    XGBoost version: %s' %xgb.__version__)
                print('    XGBoost num boost rounds = %s' %num_boost_round)
                from dask_ml.model_selection import train_test_split
                import gc
                for fold_number in range(cv_in):
                    client.run(gc.collect) 
                    objective = 'reg:squarederror'
                    params = {'objective': objective, 'max_depth': 4, 'eta': 0.01, 'subsample': 0.5, 
                                        'min_child_weight': 0.5, 'random_state':0}
                    test_size = 0.30 - (fold_number*0.05)
                    print('TS Cross Validation: %d completed' %(fold_number+1,))
                    print('    test_size = %s' %test_size)
                    X_train_fold, X_test_fold, y_train_fold, y_test_fold = train_test_split(X_train, y_train,
                                                 test_size=test_size, random_state=99)

                    ##########   Training XGBoost model using dask_xgboost #########################
                    ### the dtrain syntax can only be used xgboost 1.50 or greater. Dont use it until then.
                    #dtrain = xgb.dask.DaskDMatrix(client, X_train, y_train, enable_categorical=True)
                    
                    bst = dask_xgboost.train(client, params, X_train_fold, y_train_fold, num_boost_round=num_boost_round)                    
                    
                    bst_models.append(bst)
                    forecast_df = dask_xgboost.predict(client, bst, X_test_fold).compute()
                    forecast_df_folds.append(forecast_df)

                    ### Now compare the actuals with predictions ######
                    y_pred = forecast_df[:]
                    ### y_pred is already an array - so keep that in mind!
                    
                    concatenated = pd.DataFrame(np.c_[y_test_fold, y_pred], columns=['original', 'predicted'],index=y_test_fold.index)

                    if fold_number == 0:
                        extra_concatenated = copy.deepcopy(concatenated)
                    else:
                        extra_concatenated = extra_concatenated.append(concatenated)

                    rmse_fold, rmse_norm = print_dynamic_rmse(concatenated['original'].values, concatenated['predicted'].values,
                                                concatenated['original'].values)

                    rmse_folds.append(rmse_fold)
                    norm_rmse_folds.append(rmse_norm)
        else:
            #### If cross validation is zero, then it means cross validation needs to be skipped
            if  cv_in == 0:
                print('Skipping cross validation steps since cross_validation = %s' %cv_in)
                try:
                    model_name = 'XGBoost'
                    model = XGBRegressor(n_estimators=400, verbosity=0, random_state=0)
                except:
                    model_name = 'Random Forest'
                    model = RandomForestRegressor(n_estimators=200, random_state=99)
            else:
                for fold_number, (train_index, test_index) in enumerate(cv.split(dft)):
                    dftx = dft.head(len(train_index)+len(test_index))
                    train_fold = dftx.head(len(train_index)) ## now train will be the first segment of dftx
                    test_fold = dftx.tail(len(test_index)) ### now test will be right after train in dftx

                    horizon = len(test_fold)

                    print(f"\nFold Number: {fold_number+1} --> Train Shape: {train_fold.shape[0]} Test Shape: {test_fold.shape[0]}")

                    #########################################
                    #### Define the model with fold data ####
                    #########################################
                    ## If XGBoost is present in machine, use it. Otherwise use RandomForestRegressor
                    try:
                        model_name = 'XGBoost'
                        model = XGBRegressor(n_estimators=400, verbosity=0, random_state=0)
                    except:
                        model_name = 'Random Forest'
                        model = RandomForestRegressor(n_estimators=200, random_state=99)

                    if type(dft) == dask.dataframe.core.DataFrame:
                        nums = int(0.9*len(train_index))
                    else:
                        nums = int(0.9*train_fold.shape[0])

                    if nums <= 1:
                        nums = 2
                    ############################################
                    #### Fit the model with train_fold data ####
                    ############################################

                    X_train_fold, y_train_fold = train_fold[:nums][self.transformed_preds], train_fold[:nums][self.transformed_target]
                    X_test_fold, y_test_fold = train_fold[nums:][self.transformed_preds], train_fold[nums:][self.transformed_target]
                    if model_name == 'XGBoost':
                        model.fit(X_train_fold, y_train_fold,
                            eval_set=[(X_train_fold, y_train_fold), (X_test_fold, y_test_fold)],
                            early_stopping_rounds=50,verbose=False,
                            )
                    else:
                        ## This is for Random Forest regression
                        model.fit(X_train_fold, y_train_fold)
                    #################################################
                    #### Predict using model with test_fold data ####
                    #################################################

                    forecast_df = model.predict(test_fold[self.transformed_preds])
                    forecast_df_folds.append(forecast_df)

                    ### Now compare the actuals with predictions ######
                    y_pred = forecast_df[-horizon:]
                    ### y_pred is already an array - so keep that in mind!

                    concatenated = pd.DataFrame(np.c_[test_fold[self.transformed_target].values,
                                y_pred], columns=['original', 'predicted'],index=test_fold.index)

                    if fold_number == 0:
                        extra_concatenated = copy.deepcopy(concatenated)
                    else:
                        extra_concatenated = extra_concatenated.append(concatenated)

                    rmse_fold, rmse_norm = print_dynamic_rmse(concatenated['original'].values, concatenated['predicted'].values,
                                                concatenated['original'].values)

                    print('Cross Validation window: %d completed' %(fold_number+1,))
                    rmse_folds.append(rmse_fold)
                    norm_rmse_folds.append(rmse_norm)
                ######################################################
                ### This is where you consolidate the CV results #####
                ######################################################
        #######  Now plot feature importances ##################################
        try:
            if type(y_trues) == dask.dataframe.core.DataFrame or type(y_trues) == dask.dataframe.core.Series:
                model_name = 'dask_xgboost'
                print('\n%s-fold final RMSE (expanding Window Cross Validation):' %cv_in)
                rows = 1
                colus = 2
                fig, ax = plt.subplots(rows, colus)
                fig.set_size_inches(min(colus*8,20),rows*6)
                fig.subplots_adjust(hspace=0.3) ### This controls the space betwen rows
                fig.subplots_adjust(wspace=0.3) ### This controls the space between columns
                counter = 0
                if rows == 1:
                    ax = ax.reshape(-1,1).T
                xgb.plot_importance(bst_models[0], height=0.8, max_num_features=10, ax=ax[0][0])
                extra_concatenated.plot(ax=ax[0][1], title='ML expanding window preds vs. actuals')
                print_dynamic_rmse(extra_concatenated['original'], extra_concatenated['predicted'], 
                                extra_concatenated['original'], True)
            else:
                _ = plot_importance(model, height=0.9,importance_type='gain', title='%s Feature Importance by Gain' %model_name)
                
                cv_micro, cv_micro_pct = print_ts_model_stats(extra_concatenated['original'], extra_concatenated['predicted'], model_name)

                print('Average CV RMSE of all predictions (micro) = %0.5f' %cv_micro)
            #print('Normalized RMSE (as Std Dev of Actuals - micro) = %0.0f%%' %cv_micro_pct)
        except:
            print('Could not plot ML results due to error. Continuing...')
        ###############################################
        #### Refit the model on the entire dataset ####
        ###############################################
        print('\nFitting model on entire train set. Please be patient...')
        
        start_time = time.time()
        # Refit Model on entire train dataset (earlier, we only trained the model on the individual splits)
        # Convert to supervised learning problem

        if type(X_train) == dask.dataframe.core.DataFrame or type(X_train) == dask.dataframe.core.Series:
            ### check available memory and allocate at least 1GB of it in the Client in DASK #############################
            memory_free = str(max(1, int(psutil.virtual_memory()[0]/1e9)))+'GB'
            print('    Using Dask XGBoost algorithm with %s virtual CPUs and %s memory limit...' %(get_cpu_worker_count(), memory_free))
            client = Client(n_workers=get_cpu_worker_count(), threads_per_worker=1, processes=True, silence_logs=50,
                            memory_limit=memory_free)
            print('    Dask client configuration: %s' %client)
            print('    XGBoost version: %s' %xgb.__version__)
            bst = dask_xgboost.train(client, params, X_train, y_train, num_boost_round=num_boost_round) 
            self.model = bst 
            #X_train = X_train.head(len(X_train)) ## this converts it to a pandas dataframe
            self.df_train_prepend = ts_df.compute()[-self.lags:]
        else:
            self.model = model
            self.model.fit(X_train, y_train)
            # Save last `self.lags` which will be used for predictions later
            self.df_train_prepend = ts_df[-self.lags:]

        print('    Time taken to train model (in seconds) = %0.0f' %(time.time()-start_time))
        # # This is the new method without the leakage
        # # Drop the y value
        # test_orig_df_pred_only = test_orig_df.drop(self.original_target_col, axis=1, inplace=False)
        # forecast = self.predict(testdata=test_orig_df_pred_only, simple=False)

        # rmse, norm_rmse = print_dynamic_rmse(
        #     y_test.values,
        #     forecast['mean'],
        #     y_train.values
        # )

        # print(f"RMSE Folds: {rmse_folds}")
        # print(f"Norm RMSE Folds: {norm_rmse_folds}")

        # return self.model, forecast['mean'], rmse, norm_rmse
        return self.model, forecast_df_folds, rmse_folds, norm_rmse_folds

    def order_df(self, ts_df: pd.DataFrame) :
        """
        Given a dataframe (original), this will order the columns with
        target as the first column and the predictors as the next set of columns
        The actual target being placed in the first column is not important.
        What is important is that the order of columns is maintained consistently
        throughout the lifecycle as this will also be used when predicting with
        test data (i.e. we need to prepend some of the train data to the test data
        before transforming the original time series dataframe into a supervised
        learning problem.)
        """
        return ts_df[[self.original_target_col] + self.original_preds]


    def df_to_supervised(
        self,
        ts_df: pd.DataFrame,
        drop_zero_var: bool = False):
        """
        :param ts_df: The time series dataframe that needs to be converted
        into a supervised learning problem.
        drop_zero_var: Flag to set whether to drop zero valued rows in dataset.

        Return:
        -> Tuple[pd.DataFrame, str, List[str]]
        """
        if self.lags <= 1:
            n_in = 1
        elif self.lags >= 4:
            n_in = 4
        else:
            n_in = 1

        self.lags = copy.deepcopy(n_in)
        
        dfxs, transformed_target_name, _ = convert_timeseries_dataframe_to_supervised(
            ts_df[self.original_preds+[self.original_target_col]],
            self.original_preds+[self.original_target_col],
            self.original_target_col,
            n_in=n_in, n_out=0, dropT=False
                            )
        
        # Append the time series features (derived from the time series index)
        dfxs = create_ts_features_dask(df=dfxs, tscol=self.ts_column, drop_zero_var=False, return_original=True)
        self.transformed_target = transformed_target_name

        # Overwrite with new ones

        # transformed_pred_names = [x for x in list(dfxs) if x not in [self.transformed_target]]
        transformed_pred_names = [x for x in list(dfxs) if x not in [self.transformed_target]]

        return dfxs, transformed_target_name, transformed_pred_names

    def df_to_supervised_test(
        self,
        ts_df: pd.DataFrame,
        drop_zero_var: bool = False):
        """
        :param ts_df: The time series dataframe that needs to be converted
        into a supervised learning problem.
        drop_zero_var: Flag to set whether to drop zero valued rows in dataset.

        Return:
        -> Tuple[pd.DataFrame, str, List[str]]
        """
        if self.lags <= 1:
            n_in = 1
        elif self.lags >= 4:
            n_in = 4
        else:
            n_in = 1

        self.lags = copy.deepcopy(n_in)
        
        dfxs, transformed_target_name, _ = convert_timeseries_dataframe_to_supervised(
            ts_df[self.original_preds+[self.original_target_col]],
            self.original_preds+[self.original_target_col],
            self.original_target_col,
            n_in=n_in, n_out=0, dropT=False
                            )
        
        # Append the time series features (derived from the time series index)
        
        dfxs = create_time_series_features(dfxs, self.transformed_target, ts_column=None, drop_zero_var=False)
        self.transformed_target = transformed_target_name
        
        # Overwrite with new ones

        # transformed_pred_names = [x for x in list(dfxs) if x not in [self.transformed_target]]
        transformed_pred_names = [x for x in list(dfxs) if x not in [self.transformed_target]]

        return dfxs, transformed_target_name, transformed_pred_names


    def refit(self, ts_df: pd.DataFrame):
        """
        :param ts_df The original dataframe. All transformations to a supervised learning
        problem should be taken care internally by this method.
        'target_col': and 'lags' do not need to be passed as was the case with the fit method.
        We will simply use the values that were stored during the training process.

        Return:
        -> object
        """

        self.check_model_built()

        dfxs, _, _  = self.df_to_supervised(ts_df=ts_df, drop_zero_var=False)

        y_train = dfxs[self.transformed_target]
        X_train = dfxs[self.transformed_preds]

        self.model.fit(X_train, y_train)

        # Save last `self.lags` which will be used for predictions later
        self.df_train_prepend = ts_df[-self.lags:]

        return self


    def predict(
        self,
        testdata: Optional[pd.DataFrame]=None,
        forecast_period: Optional[int] = None,
        simple: bool = True,
        time_interval: str = '',
        ):
        """
        Return the predictions
        :param: testdata The test dataframe in pretransformed format
        :param: forecast_period Not used this this case since for ML based models,
        X_exogen is a must, hence we can use the number of rows in X_exogen
        to get the forecast period.

        Return:
        -> pandas Dataframe
        """
        
        print('For large datasets: ML predictions will take time since it has to predict each row and use that for future predictions...')
        testdata = copy.deepcopy(testdata)
        testdata_orig = copy.deepcopy(testdata)
        self.check_model_built()

        if testdata is None:
            print(
                "You have not provided the exogenous variable in order to make the prediction. " +
                "Machine Learing based models only support multivariate time series models. " +
                "Hence predictions will not be made.")
            return None
        elif isinstance(testdata, int):
            print('(Error) Testdata must be pandas dataframe for ML model. No predictions will be made.')
            return None
        else:
            testdata = load_test_data(testdata, ts_column=self.ts_column, sep=self.sep, 
                            target=self.transformed_target, dask_xgboost_flag=self.dask_xgboost_flag)

        #######   Make sure you SAVE the original dataset's index #####
        ts_index_orig = testdata.index
        
        ##### This is where we change the ts_column into a date-time index ####
        testdata, str_format = change_to_datetime_index_test(testdata, self.ts_column)

        # Placebholder for forecasted results
        y_forecasted: List[float] = []

        # print (f"Columns before adding dummy: {testdata.columns}")
        
        ts_index = testdata.index
        #print(f"Datetime Index: {ts_index}")
        
        ### the number 4 here is based on the number of lags we have set as default which is 4
        lags_index = ts_index.shift(periods=-self.lags, freq=self.time_interval)[:self.lags]

        ts_index_shifted = lags_index.append(ts_index)
        
        if str_format and not type(testdata) == dd.core.DataFrame:
            ts_index_shifted = ts_index_shifted.strftime(str_format)
            lags_index = lags_index.strftime(str_format)

        #### lags_prepend contains the lagged dataframe (with 4 lags) ####
        #### testdata_with_dummy contains the lagged + the test data frame ###
        if self.univariate:
            #### if it is univariate, you just need to combine the two empty dataframes for test
            lags_prepend = pd.DataFrame(np.zeros(len(lags_index )), index=lags_index)
            testdata_with_dummy = pd.DataFrame(np.zeros(len(ts_index_shifted )),index=ts_index_shifted)
            # Adding dummy value for target.
            testdata_with_dummy[self.original_target_col] = np.zeros((testdata_with_dummy.shape[0],1))
            testdata_with_dummy.drop(0,axis=1,inplace=True)

            # Make sure column order is correct when adding the dummy column
            testdata_with_dummy = self.order_df(testdata_with_dummy)
            # print (f"Columns after reordering: {testdata_with_dummy.columns}")

            # Just a check to make sure user is not passing the target column to predict function.
            if self.original_target_col in testdata_with_dummy.columns:
                print("Your testdata dataframe contains the target column as well. This will be deleted for the predictions.")
                testdata_with_dummy.drop(self.original_target_col, axis=1, inplace=True)

            df_prepend = self.df_train_prepend.copy(deep=True)
            #### even the df_prepend has to be made the same index format so that it can be joined with testdata_with_dummy
            if str_format and not type(testdata) == dd.core.DataFrame:
                df_prepend.index = df_prepend.index.strftime(str_format)
            df_prepend = self.order_df(df_prepend)

            df_train_prepend = testdata_with_dummy.join(df_prepend,how='left', lsuffix='test').fillna(0)

        else:
            #### if it is multivariate, you need to add empty vars for lags_prepend before joining
            testdata_with_dummy = pd.DataFrame(np.zeros(len(ts_index_shifted )),index=ts_index_shifted)
            testdata_with_dummy[self.original_preds] = 0
            # Just a check to make sure user is not passing the target column to predict function.
            if self.original_target_col in testdata_with_dummy.columns:
                print("Your testdata dataframe contains the target column as well. This will be deleted for the predictions.")
                testdata_with_dummy.drop(self.original_target_col, axis=1, inplace=True)

            if 0 in testdata_with_dummy.columns:
                testdata_with_dummy.drop(0,axis=1, inplace=True)

            time_freq = self.time_interval
            testdata_with_dummy.index = pd.to_datetime(testdata_with_dummy.index, format=str_format)
            testdata_with_dummy = testdata_with_dummy.resample(time_freq).sum()
            self.df_train_prepend = self.df_train_prepend.resample(time_freq).sum()
            testdata = testdata.resample(time_freq).sum()
            
            ###########  You have to do this for each predictor in dataframe - could take a lot of time ###
            ###   Also this will work only for float and integer variables. Won't work for categorical or object cols
            
            copy_cols1 = self.df_train_prepend.columns.tolist()
            df_pindex = self.df_train_prepend.index
            copy_cols2 = testdata.columns.tolist()
            df_tindex = testdata.index
            ####  We need to copy all the values from testdata to dummy dataframe which has zeros currently
            for each_col in copy_cols1:
                ####  We need to copy all the values from self.df_train_prepend to dummy dataframe which has zeros
                testdata_with_dummy.loc[df_pindex, each_col] = self.df_train_prepend.loc[df_pindex, each_col]
            for each_col in copy_cols2:
                testdata_with_dummy.loc[df_tindex, each_col] = testdata.loc[df_tindex, each_col]
            
            ####  We need to fill the dummy dataframe with target values from self.df_train_prepend
            self.df_train_prepend = testdata_with_dummy.fillna(0)
        
        # STEP 1:
        # self.df_prepend has the y column as well, but testdata does not.
        # Need to add a dummy column to testdata before appending the 2 dataframes
        # However, Since we are going to depend on previous values of y column to make
        # future predictions, we can not just use all zeros for the y values
        # (especially for forrecasts beyond the 1st prediction). So we will
        # make one prediction at a time and then use that prediction to make the next prediction.
        # That way, we get the most accurate prediction without leakage of informaton.



        # STEP 2:
        # Make prediction for each row. Then use the prediction for the next row.

        # TODO: Currently Frequency is missing in the data index (= None), so we can not shift the index
        ## When this is fixed in the AutoML module, we can shift and get the future index
        ## to be in a proper time series format.
        #print("Train Prepend")
        #print(self.df_train_prepend)
        #index = self.df_train_prepend.index
        #print("Index Before")
        #print(index)
        
        #index = index.shift(testdata_with_dummy.shape[0])
        #print("Index After")
        #print(index)
        
        
        #### Leave this out since we don't want to store the entire train_df in the model
        #if type(self.train_df) == dask.dataframe.core.DataFrame:
        #    traindf = self.train_df.head(self.forecast_period) ## convert from DASK to pandas
        #    df_prepend = traindf[common_cols]
        #else:

        ### Make sure that the column orders are the same ####
        #dfxs, _, _  = self.df_to_supervised(ts_df=df_prepend, drop_zero_var=False)
        #dfxs = dfxs.tail(len(testdata_with_dummy))
        #X_test = dfxs[self.transformed_preds]

        if isinstance(testdata, pd.DataFrame) or isinstance(testdata, pd.Series):
            if self.dask_xgboost_flag:
                memory_free = str(max(1, int(psutil.virtual_memory()[0]/1e9)))+'GB'
                print('    Using Dask XGBoost algorithm with %s virtual CPUs and %s memory limit...' %(get_cpu_worker_count(), memory_free))
                client = Client(n_workers=get_cpu_worker_count(), threads_per_worker=1, processes=True, silence_logs=50,
                                memory_limit=memory_free)
                bst = self.model


        for i in np.arange(self.lags, self.df_train_prepend.shape[0]):

            # Append the last n_lags of the data to the row of the X_exogen that is being preducted
            # Note that some of this will come from the last few observations of the training data
            # and the rest will come from the last few observations of the X_exogen data.
            # print(f"Prepend shape before adding test: {df_prepend.shape}")
            
            df_prepend = self.df_train_prepend.iloc[:i+1,:]

            # print(f"Prepend shape after adding test: {df_prepend.shape}")
            # print("Prepend Dataframe")
            # print(df_prepend)

            # Convert the appended dataframe to supervised learning problem
            
            dfxs, _, _  = self.df_to_supervised_test(ts_df=df_prepend, drop_zero_var=False)

            # Select only the predictors (transformed) from here
            X_test = dfxs[self.transformed_preds]
            #print("X_test")
            #print(X_test)
            
            # Forecast
            
            if isinstance(testdata, pd.DataFrame) or isinstance(testdata, pd.Series):
                if self.dask_xgboost_flag:
                    Xtest = dd.from_pandas(X_test, npartitions=1)
                    y_forecasted_temp = dask_xgboost.predict(client, bst, Xtest).compute()
                else:
                    y_forecasted_temp = self.model.predict(X_test)  # Numpy array
            elif type(testdata) == dd.core.DataFrame: 
                if not self.dask_xgboost_flag:
                    print('    Error: You cannot make predictions on test data which is a dask df if the model was trained on a pandas df. Change test to pandas.')
                    return
                bst = self.model
                memory_free = str(max(1, int(psutil.virtual_memory()[0]/1e9)))+'GB'
                print('    Using Dask XGBoost algorithm with %s virtual CPUs and %s memory limit...' %(get_cpu_worker_count(), memory_free))
                client = Client(n_workers=get_cpu_worker_count(), threads_per_worker=1, processes=True, silence_logs=50,
                                memory_limit=memory_free)
                y_forecasted_temp = dask_xgboost.predict(client, bst, X_test).compute()
            #print(y_forecasted_temp)
            y_forecasted.append(y_forecasted_temp[-1])

            # Append the predicted value for use in next prediction
            df_prepend.iloc[-1][self.original_target_col] = y_forecasted_temp[-1]

            # Remove 1st entry as it is not needed for next round
            df_prepend = df_prepend[1:]
            
            # print("df_prepend end of loop")
            # print(df_prepend)
        ##### Here is where you collect the forecasts #####
        # y_forecasted = np.array(y_forecasted)
        
        res_frame = pd.DataFrame({'yhat': y_forecasted})
        res_frame.index = ts_index
        res_frame['mean_se'] = np.nan
        res_frame['mean_ci_lower'] = np.nan
        res_frame['mean_ci_upper'] = np.nan
        print('ML predictions completed')
        ### returns a dataframe - so make sure you capture it that way ###
        if simple:
            return res_frame[['yhat']]
        else:
            return res_frame

###############################################################################################
import dask
import dask.dataframe as dd
def create_time_series_features(dft, targets, ts_column: Optional[str]=None, drop_zero_var: bool = False):
    """
    This creates between 8 and 10 date time features for each date variable.
    The number of features depends on whether it is just a year variable
    or a year+month+day and whether it has hours and mins+secs. So this can
    create all these features using just the date time column that you send in.
    It returns the entire dataframe with added variables as output.
    """
    
    reset_index = False
    time_preds = [x for x in list(dft) if x != targets]
    dtf = copy.deepcopy(dft)
    ##### This is where we convert ts_column into datetime column and create_ts_features ###
    if not ts_column in time_preds or ts_column is None:
        ###  This means there is no time series column in dataset ##
        ### this means that ts_column is already an index ####
        tscol = dft.index.name
        dtf = create_ts_features_dask(df=dft, tscol=tscol, drop_zero_var=drop_zero_var, return_original=True)
    else:
        ####  In this case, ts_column has not been converted to an index yet###
        ### In some extreme cases, date time vars are not processed yet and hence we must fill missing values here!
        if type(dft) == dask.dataframe.core.DataFrame:
            null_nums = dtf[ts_column].isnull().sum().compute()
        else:
            null_nums = dtf[ts_column].isnull().sum()
        if  null_nums > 0:
            # missing_flag = True
            new_missing_col = ts_column + '_Missing_Flag'
            dtf[new_missing_col] = 0
            dtf.loc[dtf[ts_column].isnull(),new_missing_col]=1
            dtf[ts_column] = dtf[ts_column].fillna(method='ffill')
        if dtf[ts_column].dtype == float:
            dtf[ts_column] = dtf[ts_column].astype(int)

        ### Then continue the processing with ts_column ####
        #### This is where we find the string format of datatime variable ###
        dtf, str_format = change_to_datetime_index_test(dtf, ts_column)
        dtf[ts_column] = pd.to_datetime(dtf[ts_column],format=str_format)            
        dtf = create_ts_features(df=dtf, tscol=ts_column, drop_zero_var=drop_zero_var, return_original=True)
    
    return dtf
##############################################################################################
def create_ts_features_dask(
    df,
    tscol,
    drop_zero_var: bool = True,
    return_original: bool = True) -> pd.DataFrame:
    """
    This takes in input a DASK or pandas dataframe and a date time index - if not it will fail

    :param drop_zero_var If True, it will drop any features that have zero variance
    :type drop_zero_var bool

    :param return_original If True, it will return the original dataframe concatenated with the derived features
    else, it will just return the derived features
    :type return_original bool

    :rtype pd.DataFrame
    """
    df_org = copy.deepcopy(df)
    dt_adds = []
    
    try:
        df[tscol+'_hour'] = df.index.hour.values
        dt_adds.append(tscol+'_hour')
    except:
        print('    Error in creating hour time feature. Continuing...')
    try:
        df[tscol+'_minute'] = df.index.minute.values
        dt_adds.append(tscol+'_minute')
    except:
        print('    Error in creating minute time feature. Continuing...')
    try:
        df[tscol+'_dayofweek'] = df.index.dayofweek.values
        dt_adds.append(tscol+'_dayofweek')
        df[tscol+'_quarter'] = df.index.quarter.values
        dt_adds.append(tscol+'_quarter')
        df[tscol+'_month'] = df.index.month.values
        dt_adds.append(tscol+'_month')
        df[tscol+'_year'] = df.index.year.values
        dt_adds.append(tscol+'_year')
        df[tscol+'_dayofyear'] = df.index.dayofyear.values
        dt_adds.append(tscol+'_dayofyear')
        df[tscol+'_dayofmonth'] = df.index.day.values
        dt_adds.append(tscol+'_dayofmonth')
        df[tscol+'_weekofyear'] = df.index.weekofyear.values
        dt_adds.append(tscol+'_weekofyear')
        weekends = (df[tscol+'_dayofweek'] == 5) | (df[tscol+'_dayofweek'] == 6)
        df[tscol+'_weekend'] = 0
        df[tscol+'_weekend'] = df[tscol+'_weekend'].mask(weekends, 1)
        dt_adds.append(tscol+'_weekend')
    except:
        print('    Error in creating date time derived features. Continuing...')

    #df = df[dt_adds].fillna(0).astype(int)
    
    return df
################################################################################
def create_ts_features(
    df,
    tscol,
    drop_zero_var: bool = True,
    return_original: bool = True):
    """
    This takes in input a dataframe and a date variable.
    It then creates time series features using the pandas .dt.weekday kind of syntax.
    It also returns the data frame of added features with each variable as an integer variable.

    :param drop_zero_var If True, it will drop any features that have zero variance
    :type drop_zero_var bool

    :param return_original If True, it will return the original dataframe concatenated with the derived features
    else, it will just return the derived features
    :type return_original bool

    :rtype pd.DataFrame
    """
    df_org = copy.deepcopy(df)
    dt_adds = []
    try:
        df[tscol+'_hour'] = df[tscol].dt.hour.astype(int)
        dt_adds.append(tscol+'_hour')
    except:
        print('    Error in creating hour time feature. Continuing...')
    try:
        df[tscol+'_minute'] = df[tscol].dt.minute.astype(int)
        dt_adds.append(tscol+'_minute')
    except:
        print('    Error in creating minute time feature. Continuing...')
    try:
        df[tscol+'_dayofweek'] = df[tscol].dt.dayofweek.astype(int)
        dt_adds.append(tscol+'_dayofweek')
        df[tscol+'_quarter'] = df[tscol].dt.quarter.astype(int)
        dt_adds.append(tscol+'_quarter')
        df[tscol+'_month'] = df[tscol].dt.month.astype(int)
        dt_adds.append(tscol+'_month')
        df[tscol+'_year'] = df[tscol].dt.year.astype(int)
        dt_adds.append(tscol+'_year')
        df[tscol+'_dayofyear'] = df[tscol].dt.dayofyear.astype(int)
        dt_adds.append(tscol+'_dayofyear')
        df[tscol+'_dayofmonth'] = df[tscol].dt.day.astype(int)
        dt_adds.append(tscol+'_dayofmonth')
        df[tscol+'_weekofyear'] = df[tscol].dt.weekofyear.astype(int)
        dt_adds.append(tscol+'_weekofyear')
        weekends = (df[tscol+'_dayofweek'] == 5) | (df[tscol+'_dayofweek'] == 6)
        df[tscol+'_weekend'] = 0
        df.loc[weekends, tscol+'_weekend'] = 1
        df[tscol+'_weekend'] = df[tscol+'_weekend'].astype(int)
        dt_adds.append(tscol+'_weekend')
    except:
        print('    Error in creating date time derived features. Continuing...')

    derived = df[dt_adds].fillna(0).astype(int)

    if drop_zero_var:
        derived = derived[derived.columns[derived.describe().loc['std'] != 0]]

    # print("==========AAA============")
    # print("Derived")
    # print(derived)

    if return_original:
        df = pd.concat([df_org, derived], axis=1)
    else:
        df = derived

    # print("==========BBB============")
    # print("DF")
    # print(df)

    return df
#################################################################################
####################################################################################
import re
import pdb
import pprint
from itertools import cycle, combinations
from collections import defaultdict, OrderedDict
import copy
import time
import sys
import random
import xlrd
import statsmodels
from io import BytesIO
import base64
from functools import reduce
import copy
#######################################################################################################
def classify_features(dfte, depVar, verbose=0):
    dfte = copy.deepcopy(dfte)
    if isinstance(depVar, list):
        orig_preds = [x for x in list(dfte) if x not in depVar]
    else:
        orig_preds = [x for x in list(dfte) if x not in [depVar]]
    #################    CLASSIFY  COLUMNS   HERE    ######################
    var_df = classify_columns(dfte[orig_preds], verbose)
    #####       Classify Columns   ################
    IDcols = var_df['id_vars']
    discrete_string_vars = var_df['nlp_vars']+var_df['discrete_string_vars']
    cols_delete = var_df['cols_delete']
    bool_vars = var_df['string_bool_vars'] + var_df['num_bool_vars']
    int_vars = var_df['int_vars']
    categorical_vars = var_df['cat_vars'] + var_df['factor_vars'] + int_vars + bool_vars
    date_vars = var_df['date_vars']
    if len(var_df['continuous_vars'])==0 and len(int_vars)>0:
        continuous_vars = var_df['int_vars']
        categorical_vars = left_subtract(categorical_vars, int_vars)
        int_vars = []
    else:
        continuous_vars = var_df['continuous_vars']
    preds = [x for x in orig_preds if x not in IDcols+cols_delete+discrete_string_vars]
    if len(IDcols+cols_delete+discrete_string_vars) == 0:
        print('        No variables removed since no ID or low-information variables found in data set')
    else:
        print('        %d variable(s) removed since they were ID or low-information variables'
                                %len(IDcols+cols_delete+discrete_string_vars))
        if verbose >= 1:
            print('    List of variables removed: %s' %(IDcols+cols_delete+discrete_string_vars))
    #############  Check if there are too many columns to visualize  ################
    ppt = pprint.PrettyPrinter(indent=4)
    if verbose>=1 and len(cols_list) <= max_cols_analyzed:
        marthas_columns(dft,verbose)
    elif verbose>=1 and len(cols_list) > max_cols_analyzed:
        print('   Total columns > %d, too numerous to list.' %max_cols_analyzed)
    features_dict = dict([('IDcols',IDcols),('cols_delete',cols_delete),('bool_vars',bool_vars),('categorical_vars',categorical_vars),
                        ('continuous_vars',continuous_vars),('discrete_string_vars',discrete_string_vars),
                        ('date_vars',date_vars)])
    return features_dict
#######################################################################################################
def marthas_columns(data,verbose=0):
    """
    This program is named  in honor of my one of students who came up with the idea for it.
    It's a neat way of printing data types and information compared to the boring describe() function in Pandas.
    """
    data = data[:]
    print('Data Set Shape: %d rows, %d cols' % data.shape)
    if data.shape[1] > 30:
        print('Too many columns to print')
    else:
        print('Data Set columns info:')
        for col in data.columns:
            print('* %s: %d nulls, %d unique vals, most common: %s' % (
                    col,
                    data[col].isnull().sum(),
                    data[col].nunique(),
                    data[col].value_counts().head(2).to_dict()
                ))
        print('--------------------------------------------------------------------')
################################################################################
######### NEW And FAST WAY to CLASSIFY COLUMNS IN A DATA SET #######
################################################################################
def classify_columns(df_preds, verbose=0):
    """
    Takes a dataframe containing only predictors to be classified into various types.
    DO NOT SEND IN A TARGET COLUMN since it will try to include that into various columns.
    Returns a data frame containing columns and the class it belongs to such as numeric,
    categorical, date or id column, boolean, nlp, discrete_string and cols to delete...
    ####### Returns a dictionary with 10 kinds of vars like the following: # continuous_vars,int_vars
    # cat_vars,factor_vars, bool_vars,discrete_string_vars,nlp_vars,date_vars,id_vars,cols_delete
    """
    train = copy.deepcopy(df_preds)
    #### If there are 30 chars are more in a discrete_string_var, it is then considered an NLP variable
    max_nlp_char_size = 30
    max_cols_to_print = 30
    print('############## C L A S S I F Y I N G  V A R I A B L E S  ####################')
    print('Classifying variables in data set...')
    #### Cat_Limit defines the max number of categories a column can have to be called a categorical colum
    cat_limit = 35
    float_limit = 15 #### Make this limit low so that float variables below this limit become cat vars ###
    def add(a,b):
        return a+b
    sum_all_cols = dict()
    orig_cols_total = train.shape[1]
    #Types of columns
    cols_delete = [col for col in list(train) if (len(train[col].value_counts()) == 1
                                   ) | (train[col].isnull().sum()/len(train) >= 0.90)]
    train = train[left_subtract(list(train),cols_delete)]
    var_df = pd.Series(dict(train.dtypes)).reset_index(drop=False).rename(
                        columns={0:'type_of_column'})
    sum_all_cols['cols_delete'] = cols_delete
    var_df['bool'] = var_df.apply(lambda x: 1 if x['type_of_column'] in ['bool','object']
                        and len(train[x['index']].value_counts()) == 2 else 0, axis=1)
    string_bool_vars = list(var_df[(var_df['bool'] ==1)]['index'])
    sum_all_cols['string_bool_vars'] = string_bool_vars
    var_df['num_bool'] = var_df.apply(lambda x: 1 if x['type_of_column'] in [np.uint8,
                            np.uint16, np.uint32, np.uint64,
                            'int8','int16','int32','int64',
                            'float16','float32','float64'] and len(
                        train[x['index']].value_counts()) == 2 else 0, axis=1)
    num_bool_vars = list(var_df[(var_df['num_bool'] ==1)]['index'])
    sum_all_cols['num_bool_vars'] = num_bool_vars
    ######   This is where we take all Object vars and split them into diff kinds ###
    discrete_or_nlp = var_df.apply(lambda x: 1 if x['type_of_column'] in ['object']  and x[
        'index'] not in string_bool_vars+cols_delete else 0,axis=1)
    ######### This is where we figure out whether a string var is nlp or discrete_string var ###
    var_df['nlp_strings'] = 0
    var_df['discrete_strings'] = 0
    var_df['cat'] = 0
    var_df['id_col'] = 0
    discrete_or_nlp_vars = var_df.loc[discrete_or_nlp==1]['index'].values.tolist()
    if len(var_df.loc[discrete_or_nlp==1]) != 0:
        for col in discrete_or_nlp_vars:
            #### first fill empty or missing vals since it will blowup ###
            train[col] = train[col].fillna('  ')
            if train[col].map(lambda x: len(x) if type(x)==str else 0).mean(
                ) >= max_nlp_char_size and len(train[col].value_counts()
                        ) <= int(0.9*len(train)) and col not in string_bool_vars:
                var_df.loc[var_df['index']==col,'nlp_strings'] = 1
            elif len(train[col].value_counts()) > cat_limit and len(train[col].value_counts()
                        ) <= int(0.9*len(train)) and col not in string_bool_vars:
                var_df.loc[var_df['index']==col,'discrete_strings'] = 1
            elif len(train[col].value_counts()) > cat_limit and len(train[col].value_counts()
                        ) == len(train) and col not in string_bool_vars:
                var_df.loc[var_df['index']==col,'id_col'] = 1
            else:
                var_df.loc[var_df['index']==col,'cat'] = 1
    nlp_vars = list(var_df[(var_df['nlp_strings'] ==1)]['index'])
    sum_all_cols['nlp_vars'] = nlp_vars
    discrete_string_vars = list(var_df[(var_df['discrete_strings'] ==1) ]['index'])
    sum_all_cols['discrete_string_vars'] = discrete_string_vars
    ###### This happens only if a string column happens to be an ID column #######
    #### DO NOT Add this to ID_VARS yet. It will be done later.. Dont change it easily...
    #### Category DTYPE vars are very special = they can be left as is and not disturbed in Python. ###
    var_df['dcat'] = var_df.apply(lambda x: 1 if str(x['type_of_column'])=='category' else 0,
                            axis=1)
    factor_vars = list(var_df[(var_df['dcat'] ==1)]['index'])
    sum_all_cols['factor_vars'] = factor_vars
    ########################################################################
    date_or_id = var_df.apply(lambda x: 1 if x['type_of_column'] in [np.uint8,
                         np.uint16, np.uint32, np.uint64,
                         'int8','int16',
                        'int32','int64']  and x[
        'index'] not in string_bool_vars+num_bool_vars+discrete_string_vars+nlp_vars else 0,
                                        axis=1)
    ######### This is where we figure out whether a numeric col is date or id variable ###
    var_df['int'] = 0
    var_df['date_time'] = 0
    ### if a particular column is date-time type, now set it as a date time variable ##
    var_df['date_time'] = var_df.apply(lambda x: 1 if x['type_of_column'] in ['<M8[ns]','datetime64[ns]']  and x[
        'index'] not in string_bool_vars+num_bool_vars+discrete_string_vars+nlp_vars else 0,
                                        axis=1)
    ### this is where we save them as date time variables ###
    if len(var_df.loc[date_or_id==1]) != 0:
        for col in var_df.loc[date_or_id==1]['index'].values.tolist():
            if len(train[col].value_counts()) == len(train):
                if train[col].min() < 1900 or train[col].max() > 2050:
                    var_df.loc[var_df['index']==col,'id_col'] = 1
                else:
                    try:
                        pd.to_datetime(train[col],infer_datetime_format=True)
                        var_df.loc[var_df['index']==col,'date_time'] = 1
                    except:
                        var_df.loc[var_df['index']==col,'id_col'] = 1
            else:
                if train[col].min() < 1900 or train[col].max() > 2050:
                    if col not in num_bool_vars:
                        var_df.loc[var_df['index']==col,'int'] = 1
                else:
                    try:
                        pd.to_datetime(train[col],infer_datetime_format=True)
                        var_df.loc[var_df['index']==col,'date_time'] = 1
                    except:
                        if col not in num_bool_vars:
                            var_df.loc[var_df['index']==col,'int'] = 1
    else:
        pass
    int_vars = list(var_df[(var_df['int'] ==1)]['index'])
    date_vars = list(var_df[(var_df['date_time'] == 1)]['index'])
    id_vars = list(var_df[(var_df['id_col'] == 1)]['index'])
    sum_all_cols['int_vars'] = int_vars
    copy_date_vars = copy.deepcopy(date_vars)
    for date_var in copy_date_vars:
        #### This test is to make sure sure date vars are actually date vars
        try:
            pd.to_datetime(train[date_var],infer_datetime_format=True)
        except:
            ##### if not a date var, then just add it to delete it from processing
            cols_delete.append(date_var)
            date_vars.remove(date_var)
    sum_all_cols['date_vars'] = date_vars
    sum_all_cols['id_vars'] = id_vars
    sum_all_cols['cols_delete'] = cols_delete
    ## This is an EXTREMELY complicated logic for cat vars. Don't change it unless you test it many times!
    var_df['numeric'] = 0
    float_or_cat = var_df.apply(lambda x: 1 if x['type_of_column'] in ['float16',
                            'float32','float64'] else 0,
                                        axis=1)
    if len(var_df.loc[float_or_cat == 1]) > 0:
        for col in var_df.loc[float_or_cat == 1]['index'].values.tolist():
            if len(train[col].value_counts()) > 2 and len(train[col].value_counts()
                ) <= float_limit and len(train[col].value_counts()) <= len(train):
                var_df.loc[var_df['index']==col,'cat'] = 1
            else:
                if col not in num_bool_vars:
                    var_df.loc[var_df['index']==col,'numeric'] = 1
    cat_vars = list(var_df[(var_df['cat'] ==1)]['index'])
    continuous_vars = list(var_df[(var_df['numeric'] ==1)]['index'])
    ########  V E R Y    I M P O R T A N T   ###################################################
    ##### There are a couple of extra tests you need to do to remove abberations in cat_vars ###
    cat_vars_copy = copy.deepcopy(cat_vars)
    for cat in cat_vars_copy:
        if df_preds[cat].dtype==float:
            continuous_vars.append(cat)
            cat_vars.remove(cat)
            var_df.loc[var_df['index']==cat,'cat'] = 0
            var_df.loc[var_df['index']==cat,'numeric'] = 1
        elif len(df_preds[cat].value_counts()) == df_preds.shape[0]:
            id_vars.append(cat)
            cat_vars.remove(cat)
            var_df.loc[var_df['index']==cat,'cat'] = 0
            var_df.loc[var_df['index']==cat,'id_col'] = 1
    sum_all_cols['cat_vars'] = cat_vars
    sum_all_cols['continuous_vars'] = continuous_vars
    sum_all_cols['id_vars'] = id_vars
    ###### This is where you consoldate the numbers ###########
    var_dict_sum = dict(zip(var_df.values[:,0], var_df.values[:,2:].sum(1)))
    for col, sumval in var_dict_sum.items():
        if sumval == 0:
            print('%s of type=%s is not classified' %(col,train[col].dtype))
        elif sumval > 1:
            print('%s of type=%s is classified into more then one type' %(col,train[col].dtype))
        else:
            pass
    ###############  This is where you print all the types of variables ##############
    ####### Returns 8 vars in the following order: continuous_vars,int_vars,cat_vars,
    ###  string_bool_vars,discrete_string_vars,nlp_vars,date_or_id_vars,cols_delete
    ##### now collect all the column types and column names into a single dictionary to return!
    len_sum_all_cols = reduce(add,[len(v) for v in sum_all_cols.values()])
    if len_sum_all_cols == orig_cols_total:
        print('    %d Predictors classified...' %orig_cols_total)
        #print('        This does not include the Target column(s)')
    else:
        print('No of columns classified %d does not match %d total cols. Continuing...' %(
                   len_sum_all_cols, orig_cols_total))
        ls = sum_all_cols.values()
        flat_list = [item for sublist in ls for item in sublist]
        if len(left_subtract(list(train),flat_list)) == 0:
            print(' Missing columns = None')
        else:
            print(' Missing columns = %s' %left_subtract(list(train),flat_list))
    return sum_all_cols
###############################################################################################
def left_subtract(l1,l2):
    lst = []
    for i in l1:
        if i not in l2:
            lst.append(i)
    return lst
#################################################################################
import copy
def create_univariate_lags_for_train(df, vals, each_lag):
    df = copy.deepcopy(df)
    df['lag_' + str(each_lag)+"_"+ str(vals)] = df[vals].shift(each_lag)
    return df.fillna(0)
################################################################################
import copy
def create_univariate_lags_for_test(test, train, vals, each_lag):
    test = copy.deepcopy(test)
    max_length = min((len(train), len(test)))
    new_col = ['lag_' + str(each_lag)+"_"+ str(vals)]
    new_list = []
    for i in range(max_length):
         new_list.append(train[vals][:].iloc[-each_lag+i])
    test[new_col] = 0
    try:
        test.loc[:max_length,new_col] = np.array(new_list)
    except:
        test.loc[:max_length,new_col] = np.array(new_list).reshape(-1,1)
    return test
################################################################################
import multiprocessing
def get_cpu_worker_count():
    return multiprocessing.cpu_count()
#############################################################################################
