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
import xgboost as xgb
from dask.distributed import Client, progress
import psutil
import json
################################################################################################
from .build_base import BuildBase
from .ml_models import complex_XGBoost_model, data_transform, analyze_problem_type

# helper functions
from ..utils import print_static_rmse, print_dynamic_rmse, convert_timeseries_dataframe_to_supervised, print_ts_model_stats
from ..utils import change_to_datetime_index, change_to_datetime_index_test, reduce_mem_usage, load_test_data
from ..utils import My_LabelEncoder, My_LabelEncoder_Pipe
from ..utils import left_subtract
#################################################################################################
import pdb
import time
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin #gives fit_transform method for free
import pdb
from sklearn.base import TransformerMixin
from collections import defaultdict
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import FunctionTransformer
###################################################################################################
class BuildML(BuildBase):
    def __init__(self, scoring: str = '', forecast_period: int = 2, ts_column: str = '', 
                        time_interval: str = '', sep: str = ',', dask_xgboost_flag: int = 0,
                        strf_time_format: str = '', num_boost_rounds = 250, verbose: int = 0):
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
        print("### Be careful setting dask_xgboost_flag to True since dask is unstable and doesn't work sometime's ###")
        self.dask_xgboost_flag = dask_xgboost_flag    
        self.problem_type: str = "Regression"
        self.multilabel: bool = False
        self.transformed_target: str = ""
        self.transformed_preds: List[str] = []
        self.scaler = StandardScaler()
        self.num_boost_rounds = num_boost_rounds
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
        self.original_preds = [x for x in list(ts_df) if x not in self.original_target_col]
        self.ts_column = ts_column
        ts_index = ts_df.index 
        
        ############     This is where we check if this is a univariate or multivariate problem ########
        if len(self.original_preds) > 0:
            ####################     This is for multivariate problems only  ##########################
            self.univariate = False
            if type(ts_df) == dd.core.DataFrame or type(ts_df) == dd.core.Series:
                #### This is for dask dataframes for multivariate problems #############################
                continuous_vars = ts_df.select_dtypes('number').columns.tolist()
                numvars = [x for x in continuous_vars if x not in self.original_target_col]
                catvars = ts_df.select_dtypes('object').columns.tolist() + ts_df.select_dtypes('category').columns.tolist()
                preds = [x for x in list(ts_df) if x not in self.original_target_col+catvars]
                if len(catvars) > 0:
                    print('    Warning: Dropping Categorical variables %s. You can Label Encode them and try ML modeling again...' %catvars)
            else:
                ########      This is for pandas dataframes  only     ##########################################
                features_dict = classify_features(ts_df, self.original_target_col)
                idcols = features_dict['IDcols']
                datevars = features_dict['date_vars']
                cols_to_remove = features_dict['cols_delete'] + features_dict['IDcols'] + features_dict['discrete_string_vars']
                preds = [x for x in list(ts_df) if x not in self.original_target_col+cols_to_remove]
                #catvars = ts_df[preds].select_dtypes(include = 'object').columns.tolist() + ts_df[preds].select_dtypes(include = 'category').columns.tolist()
                #numvars = ts_df[preds].select_dtypes(include = 'number').columns.tolist()
                numvars = features_dict['continuous_vars']
                catvars = features_dict['categorical_vars']
                ########  This is only for pandas dataframes  ##########################################
                if len(catvars) > 0:
                    print('    We will convert %s Categorical variables to numeric using a Transformer pipeline...' %len(catvars))
                self.original_preds = numvars + catvars
                preds = numvars+catvars
                if len(numvars) > 30:
                    print('    Warning: %s numeric variables. Hence too many lag features will be generated. Set lag to 3 or less...' %len(numvars))

            #######    This is where we set up the predictors to use for multivariate forecasting   #####################
            self.original_preds = preds
            if len(preds) > 30:
                print('    Warning: too many continuous variables = %s . Hence set lag=2 in "setup" to avoid excessive feature generation...' %len(numvars))
        else:
            #### if there is only one variable and that is the target then it is a univariate problem ##########
            self.univariate = True
            preds = self.original_preds[:]

        ts_df = ts_df[preds+self.original_target_col]
        self.problem_type, self.multilabel = analyze_problem_type(ts_df, self.original_target_col, verbose=1)

        # Order data
        ts_df = self.order_df(ts_df)
        
        if type(ts_df) == dd.core.DataFrame or type(ts_df) == dd.core.Series:
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
            cv_in = 2
        NFOLDS = self.get_num_folds_from_cv(cv_in)
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
        dft = dfxs[self.transformed_preds+self.transformed_target]

        if self.scoring == '':
            self.scoring = 'neg_root_mean_squared_error'
        elif self.scoring == 'rmse':
            self.scoring = 'neg_root_mean_squared_error'

        print('\nRunning Cross Validation using XGBoost model..')
        ###########################################################################################
        if type(dft) == dd.core.DataFrame or type(dft) == dd.core.Series:
            test_size = 0.1
            max_trainsize = 0.9
        else:
            max_trainsize = len(dft) - self.forecast_period
            test_size = max(self.forecast_period, int(0.1*dft.shape[0]))
        try:
            cv = TimeSeriesSplit(n_splits=NFOLDS, test_size=test_size) ### this works only sklearn v 0.0.24]
        except:
            cv = TimeSeriesSplit(n_splits=NFOLDS, max_train_size = max_trainsize)

        print('    Max. iterations using expanding window cross validation = %d' %NFOLDS)
        start_time = time.time()
        rmse_folds = []
        norm_rmse_folds = []
        concatenated = pd.DataFrame()
        extra_concatenated = []
        bst_models = []
        important_features = []
        #########################################################################################################
        ################    M  O  D  E  L    H Y P E R   P A R A M    T U N I N G    T A K E S   P L A C E   ####
        ######################################################################################################### 
        if type(dft) == dd.core.DataFrame or type(dft) == dd.core.Series:
            #################  This is for DASK DATAFRAMES.  We use DASK_XGBOOST here  ###
            if  cv_in == 0:
                print('Model training performed only once since cross_validation = %s' %cv_in)
                cv_in = 1
            ### In case there is no cross validation, just run it once #################################
            ### check available memory and allocate at least 1GB of it in the Client in DASK #############################
            memory_free = str(max(1, int(psutil.virtual_memory()[0]/1e9)))+'GB'
            print('    Using Dask XGBoost algorithm with %s virtual CPUs and %s memory limit...' %(get_cpu_worker_count(), memory_free))
            client = Client(n_workers=get_cpu_worker_count(), threads_per_worker=1, processes=True, silence_logs=50,
                            memory_limit=memory_free)
            print('    Dask client configuration: %s' %client)
            print('    XGBoost version: %s' %xgb.__version__)
            import gc
            for fold_number in range(cv_in):
                client.run(gc.collect) 
                objective = 'reg:squarederror'
                params = {'objective': objective, 'max_depth': 4, 'eta': 0.01, 'subsample': 0.5, 
                                    'min_child_weight': 0.5, 'random_state':0}
                test_size = 0.30 - (fold_number*0.05)
                train_fold, test_fold = dft.random_split([1-test_size, test_size], random_state=9999, shuffle=False)
                X_train_fold, y_train_fold = train_fold[self.transformed_preds], train_fold[self.transformed_target]
                X_test_fold, y_test_fold = test_fold[self.transformed_preds], test_fold[self.transformed_target]
                print('train fold shape %s, test fold shape = %s' %(X_train_fold.shape, X_test_fold.shape))
                
                ########################################################################################
                ##########   Training XGBoost model using xgboost version 1.5.1 or greater #############
                ### the dtrain syntax can only be used xgboost 1.50 or greater. Dont use it until then.
                ########################################################################################
                dtrain = xgb.dask.DaskDMatrix(client, X_train_fold, y_train_fold, enable_categorical=False, feature_names=self.transformed_preds)
                #### SYNTAX BELOW WORKS WELL. BUT YOU CANNOT DO CV or EVALS WITH DASK XGBOOST AS OF NOW ####
                print("### number of booster rounds = %s which can be set during setup ###" %self.num_boost_rounds)
                bst = xgb.dask.train(client, params, dtrain, num_boost_round=self.num_boost_rounds)
                bst_models.append(bst['booster'])
                dtest = xgb.dask.DaskDMatrix(client, X_test_fold, enable_categorical=False, feature_names=self.transformed_preds)
                forecast_df = xgb.dask.predict(client, bst['booster'], dtest)
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
                print('TS Cross Validation: %d completed' %(fold_number+1,))

                rmse_folds.append(rmse_fold)
                norm_rmse_folds.append(rmse_norm)

            #######     This is for DASK Dataframes only     ######################
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
            extra_concatenated.plot(ax=ax[0][1], title='%s expanding window preds vs. actuals' %model_name)
            print_dynamic_rmse(extra_concatenated['original'], extra_concatenated['predicted'], 
                            extra_concatenated['original'], True)
        else:
            ####################### This is for Pandas Dataframes only ##########
            X_train, y_train = dft[self.transformed_preds], dft[self.transformed_target]
            ######  This is for regular pandas dataframes only  ##############
            nums = int(0.9*dft.shape[0])
            if nums <= 1:
                nums = 2
            ############################################
            #### Fit the model with train_fold data ####
            ############################################

            X_train_fold, y_train_fold = X_train[:nums], y_train[:nums]
            X_test_fold, y_test_fold = X_train[nums:], y_train[nums:]
            print('train fold shape %s, test fold shape = %s' %(X_train_fold.shape, X_test_fold.shape))
            
            model_name = 'XGBoost'
            print('### Number of booster rounds = %s for XGBoost which can be set during setup ####' %self.num_boost_rounds)
            outputs = complex_XGBoost_model(X_train_fold,y_train_fold,
                        X_test_fold, log_y=False, GPU_flag=False,
                        scaler='', enc_method='', n_splits=cv_in, 
                        num_boost_round=self.num_boost_rounds, verbose=0)
            print('XGBoost model tuning completed')

            ###### always the last output is model  and the first output is predictions ######
            model = outputs[-1]
            y_pred = outputs[0]
            self.scaler = outputs[1]
            ############## Print results for each target one by one ################
            if len(self.original_target_col): 
                ### just make sure that there is at least one column in predictions ####
                y_pred = y_pred.reshape(-1,1)
            for each_i, each_target in enumerate(self.original_target_col):
                print('Target = %s...CV results:' %each_target)
                concatenated = pd.DataFrame(np.c_[y_test_fold.iloc[:,each_i], y_pred[:,each_i]], columns=['actual', 'predicted'],index=y_test_fold.index)
                rmse_fold, rmse_norm = print_dynamic_rmse(concatenated['actual'].values, concatenated['predicted'].values,
                                            concatenated['actual'].values)
                rmse_folds.append(rmse_fold)
                norm_rmse_folds.append(rmse_norm)
                forecast_df_folds.append(y_pred)
                extra_concatenated.append(concatenated)
            #######  Now plot feature importances for pandas dataframes ###########
            try:
                #####  This is for plotting pandas dataframes only ################
                rows = len(self.original_target_col)
                colus = 2
                fig, ax = plt.subplots(rows, colus)
                fig.set_size_inches(min(colus*8,20),rows*6)
                fig.subplots_adjust(hspace=0.3) ### This controls the space betwen rows
                fig.subplots_adjust(wspace=0.3) ### This controls the space between columns
                counter = 0
                if rows == 1:
                    ax = ax.reshape(-1,1).T
                ###### Now multi_labels have different kind of models from single labels  ################
                if self.multilabel:
                    ##### This is for multi_label models ###########
                    for each_i in range(len(self.original_target_col)):
                        _ = plot_importance(model.estimators_[each_i], height=0.9,importance_type='gain', 
                                    title='%s Feature Importance by Gain' %self.original_target_col[each_i],
                                    max_num_features=10, ax=ax[each_i][0])
                        extra_concatenated[each_i].plot(ax=ax[each_i][1], title='%s expanding window preds vs. actuals' %self.original_target_col[each_i])
                else:
                    ##### This is for single_label models ###########
                    for each_i in range(len(self.original_target_col)):
                        _ = plot_importance(model, height=0.9,importance_type='gain', 
                                    title='%s Feature Importance by Gain' %self.original_target_col[0],
                                    max_num_features=10, ax=ax[each_i][0])
                        extra_concatenated[each_i].plot(ax=ax[each_i][1], title='%s expanding window preds vs. actuals' %self.original_target_col[each_i])
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
        
        if type(X_train) == dd.core.DataFrame or type(X_train) == dd.core.Series:
            ### check available memory and allocate at least 1GB of it in the Client in DASK #############################
            memory_free = str(max(1, int(psutil.virtual_memory()[0]/1e9)))+'GB'
            print('    Using Dask XGBoost algorithm with %s virtual CPUs and %s memory limit...' %(get_cpu_worker_count(), memory_free))
            client = Client(n_workers=get_cpu_worker_count(), threads_per_worker=1, processes=True, silence_logs=50,
                            memory_limit=memory_free)
            print('    Dask client configuration: %s' %client)
            print('    XGBoost version: %s' %xgb.__version__)
            model = bst['booster']
            dtrain = xgb.dask.DaskDMatrix(client, X_train, y_train, enable_categorical=False, feature_names=self.transformed_preds)
            #### SYNTAX BELOW WORKS WELL. BUT YOU CANNOT DO CV or EVALS WITH DASK XGBOOST AS OF NOW ####
            params = json.loads(model.save_config())
            extra = {'num_class': 1}                 
            params.update(extra)                 
            trained = xgb.dask.train(client, params, dtrain, num_boost_round=self.num_boost_rounds, xgb_model=model)
            self.model = trained['booster']
            #X_train = X_train.head(len(X_train)) ## this converts it to a pandas dataframe
            self.df_train_prepend = ts_df.compute()[-self.lags:]
        else:
            if str(model).split(".")[0] == '<xgboost':
                ### if this is a booster-type model, then you have to do this ###
                ##### Once you get the model back since it was trained on a scaled dataset, you have to scale it again ###
                X_train = pd.DataFrame(self.scaler.transform(X_train), columns = X_train.columns)
                dtrain = xgb.DMatrix(X_train, label=y_train)
                params = json.loads(model.save_config())
                extra = {'num_class': 1}
                params.update(extra)
                trained = xgb.train(params, dtrain, xgb_model=model)
                self.model = trained
            else:
                ### if this is a regular sklearn type syntax model, then use this syntax ##
                self.model = model
                self.model.fit(X_train, y_train)
                # Save last `self.lags` which will be used for predictions later
            self.df_train_prepend = ts_df[-self.lags:]        
        #print('After training completed. Full forecast on train again:\n%s' %self.model.predict(dtrain))
        print('    Time taken to train model (in seconds) = %0.0f' %(time.time()-start_time))
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
        return ts_df[self.original_target_col + self.original_preds]


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
        elif self.lags >= 10:
            n_in = 10
        else:
            n_in = 1

        self.lags = copy.deepcopy(n_in)
        
        dfxs, transformed_target_name, _ = convert_timeseries_dataframe_to_supervised(
            ts_df[self.original_preds+self.original_target_col],
            self.original_preds+self.original_target_col,
            self.original_target_col,
            n_in=n_in, n_out=0, dropT=False
                            )
        
        # Append the time series features (derived from the time series index)

        dfxs = create_ts_features_dask(df=dfxs, tscol=self.ts_column, drop_zero_var=False, return_original=True)
        self.transformed_target = transformed_target_name

        # Overwrite with new ones
        if type(dfxs) == dd.core.DataFrame or type(dfxs) == dd.core.Series:
            transformed_pred_names = [x for x in dfxs.columns if x not in self.transformed_target]
        else:
            transformed_pred_names = [x for x in list(dfxs) if x not in self.transformed_target]
        
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
        n_in = self.lags
        
        dfxs, transformed_target_name, _ = convert_timeseries_dataframe_to_supervised(
            ts_df[self.original_preds+self.original_target_col],
            self.original_preds+self.original_target_col,
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
            if (self.dask_xgboost_flag and isinstance(testdata, pd.DataFrame)) or self.dask_xgboost_flag and isinstance(testdata, pd.Series):
                print('FYI: You trained the model on dask dataframes and testing it on pandas dataframes. Continuing...')
                testdata = load_test_data(testdata, ts_column=self.ts_column, sep=self.sep, 
                            target=self.transformed_target, dask_xgboost_flag=self.dask_xgboost_flag)
            else:
                testdata = load_test_data(testdata, ts_column=self.ts_column, sep=self.sep, 
                            target=self.transformed_target, dask_xgboost_flag=self.dask_xgboost_flag)
        
        #######   Make sure you SAVE the original dataset's index #####
        ts_index_orig = testdata.index
        
        ##### This is where we change the ts_column into a date-time index ####
        str_format = self.strf_time_format
        testdata, str_format = change_to_datetime_index_test(testdata, self.ts_column, str_format)

        # Placeholder for forecasted results
        y_forecasted = np.array([])

        # print (f"Columns before adding dummy: {testdata.columns}")
        
        ts_index = testdata.index
        #print(f"Datetime Index: {ts_index}")
        
        ### the number 4 here is based on the number of lags we have set as default which is 4
        lags_index = ts_index.shift(periods=-self.lags, freq=self.time_interval)[:self.lags]

        ts_index_shifted = lags_index.append(ts_index)
        
        if str_format and not type(testdata) == dd.core.DataFrame:
            ts_index_shifted = ts_index_shifted.strftime(str_format)
            lags_index = lags_index.strftime(str_format)

        ################## This applies to both Univariate and Multivariate problems   ###############
        ####  ##########       You need to do this one step at a time.  ##############################
        ####  you need to iterate on the entire testdata one row at a time to make predictions     ###
        ####  First start by adding one row from testdata to df_train_prepend to create df_pre_test ##
        ####  Then transform df_pre_test into a supervised ML dataset, to become df_post_test      ###
        ####  Then use xgb model to predict last row of this df_post_test. You will get one value. ###
        ####  Add this forecast to df_pre_test in its last row's target. df_pre_test is now ready. ###
        ####  Now begin at the top of cycle again by adding one row to df_pre_test and continue... ###  
        ###########  Beginning of the cycle of forecasts for test data ###############################
        
        index_name = self.df_train_prepend.index.name
        df_pre_test = copy.deepcopy(self.df_train_prepend)
        iter_limit = testdata.shape[0]
        for i in range(iter_limit):
            one_row_from_test = testdata.iloc[i,:]
            one_row_from_test = pd.DataFrame(one_row_from_test).T
            one_row_from_test = one_row_from_test.infer_objects()
            df_pre_test = pd.concat([df_pre_test, one_row_from_test], axis=0)
            df_pre_test.index.name = index_name
            df_post_test, _, _  = self.df_to_supervised(ts_df=df_pre_test.fillna(0), drop_zero_var=False)
            df_post_test = df_post_test.infer_objects()

            if len(left_subtract(df_post_test.columns.tolist(),self.original_target_col) ) != len(df_post_test.columns.tolist()):
                one_row_to_predict = df_post_test.drop(self.original_target_col, axis=1)
                if one_row_to_predict.shape[0] > 1:
                    ### If you have only one row, there is no need to select the last row ##
                    one_row_to_predict = one_row_to_predict.iloc[-1,:]
            else:
                one_row_to_predict = df_post_test.iloc[-1,:][self.transformed_preds]
           
            if one_row_to_predict.shape[0] > 1 :
                X_test = pd.DataFrame(one_row_to_predict).T
            else:
                 #### If there is only one row to predict => just leave it as it is ######
                X_test = copy.deepcopy(one_row_to_predict)
            X_test = X_test.infer_objects()
            
            if isinstance(testdata, pd.DataFrame) or isinstance(testdata, pd.Series):
                if self.dask_xgboost_flag:
                    dtest = xgb.DMatrix(X_test)
                    y_forecasted_temp = self.model.predict(dtest)
                else:
                    if str(self.model).split(".")[0] == '<xgboost':
                        X_test = pd.DataFrame(self.scaler.transform(X_test), columns = X_test.columns)
                        X_test = xgb.DMatrix(X_test)
                    y_forecasted_temp = self.model.predict(X_test)  # Numpy array
                if len(self.original_target_col) == 1:
                    ### you need to reshape it so that later functions can work #####
                    y_forecasted_temp = y_forecasted_temp.reshape(-1,1)
            elif type(testdata) == dd.core.DataFrame or type(testdata) == dd.core.Series: 
                if not self.dask_xgboost_flag:
                    print('    Error: You cannot make predictions on a dask_dataframe for test data if the model was not trained on dask_dataframe.')
                    return testdata
                bst = self.model
                memory_free = str(max(1, int(psutil.virtual_memory()[0]/1e9)))+'GB'
                print('    Using Dask XGBoost algorithm with %s virtual CPUs and %s memory limit...' %(get_cpu_worker_count(), memory_free))
                client = Client(n_workers=get_cpu_worker_count(), threads_per_worker=1, processes=True, silence_logs=50,
                                memory_limit=memory_free)
                Xtest = dd.from_pandas(X_test, npartitions=1)
                dtest = xgb.dask.DaskDMatrix(client, Xtest)
                y_forecasted_temp = xgb.dask.predict(client, self.model, dtest).compute()
                if len(self.original_target_col) == 1:
                    ### you need to reshape it so that later functions can work #####
                    y_forecasted_temp = y_forecasted_temp.reshape(-1,1)

            #### add this forecasted value to df_pre_test #################
            for each_i, each_target in enumerate(self.transformed_target):
                df_pre_test.iloc[-1,:][each_target] = y_forecasted_temp[:,each_i]

            ### This will work for both single-label and multi-label problems. I have tested it works. ###
            if i == 0:
                y_forecasted = copy.deepcopy(y_forecasted_temp)
            else:
                y_forecasted = np.r_[y_forecasted, y_forecasted_temp]
            #####    End of this cycle of forecasts for testdata #######################
        #####   End of all predictions ################           
        ##### Here is where you collect the forecasts #####
        # y_forecasted = np.array(y_forecasted)
        try:
            res_frame = pd.DataFrame(y_forecasted, columns=['yhat'], index=ts_index)
        except:
            ### Sometimes the index doesn't match, so better to leave out index in that case ###
            res_frame = pd.DataFrame(y_forecasted, columns=self.transformed_target)
        res_frame['mean_se'] = np.nan
        res_frame['mean_ci_lower'] = np.nan
        res_frame['mean_ci_upper'] = np.nan
        print('ML predictions completed')
        ### returns a dataframe - so make sure you capture it that way ###
        return res_frame

###############################################################################################
import dask
import dask.dataframe as dd
def create_time_series_features(dft, targets, ts_column: Optional[str]=None, 
                drop_zero_var: bool = False):
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
        print('Alert: %s has not been converted to an index yet!' %ts_column)
        ### In some extreme cases, date time vars are not processed yet and hence we must fill missing values here!
        if type(dft) == dd.core.DataFrame or type(dft) == dd.core.Series:
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
        #dtf, str_format = change_to_datetime_index_test(dtf, ts_column)
        #dtf[ts_column] = pd.to_datetime(dtf[ts_column],format=str_format)            
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
