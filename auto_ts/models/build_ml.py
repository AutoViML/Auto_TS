import warnings
from typing import List, Optional, Tuple

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from pandas.core.generic import NDFrame # type:ignore

from tscv import GapWalkForward # type: ignore

# imported ML models from scikit-learn
from sklearn.model_selection import (ShuffleSplit, StratifiedShuffleSplit, # type: ignore
                                    TimeSeriesSplit, cross_val_score) # type: ignore
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis # type: ignore
from sklearn.ensemble import (BaggingRegressor, ExtraTreesRegressor,  # type: ignore
                             RandomForestClassifier, ExtraTreesClassifier,  # type: ignore
                             AdaBoostRegressor, AdaBoostClassifier) # type: ignore
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeCV # type: ignore
from sklearn.svm import LinearSVC, SVR, LinearSVR # type: ignore
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier  # type: ignore

from .build_base import BuildBase

# helper functions
from ..utils import print_static_rmse, print_dynamic_rmse, convert_timeseries_dataframe_to_supervised
import pdb

class BuildML(BuildBase):
    def __init__(self, scoring: str = '', forecast_period: int = 2, verbose: int = 0):
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

        self.transformed_target: str = ""
        self.transformed_preds: List[str] = []

        # This saves the last `self.lags` of the original train dataframe
        # This is needed during predictions to transform the X_test
        # to a supervised learning problem.
        self.df_train_prepend: pd.DataFrame = pd.DataFrame()
       

    def fit(self, ts_df: pd.DataFrame, target_col: str, cv: Optional[int]=None, lags: int = 0):
        """
        Build a Time Series Model using Machine Learning models.
        Quickly builds and runs multiple models for a clean data set (only numerics).
        """

        self.original_target_col = target_col
        self.lags = lags
        self.original_preds = [x for x in list(ts_df) if x not in [self.original_target_col]]

        # Order data
        ts_df = self.order_df(ts_df)

        train_orig_df = ts_df[:-self.forecast_period]
        test_orig_df = ts_df[-self.forecast_period:] # This will be used for making the dynamic prediction

        dfxs, self.transformed_target, self.transformed_preds = self.df_to_supervised(ts_df)
        
        # train = dfxs[:-self.forecast_period]
        # test = dfxs[-self.forecast_period:]

        # y_train = train[self.transformed_target]
        # X_train = train[self.transformed_preds]
        
        # y_test = test[self.transformed_target]
        # X_test = test[self.transformed_preds]
        
        # print("ML Diagnostics:")
        # print(f"Lags: {self.lags}")
        # print(f"Original Shape: {ts_df.shape}")
        # print(f"Transformed Shape: {dfxs.shape}")
        # print(f"Forecast Period: {self.forecast_period}")
        # print(f"Train Shape: {X_train.shape}")
        # print(f"Test Shape: {X_test.shape}")
        # print(f"Train Index: {X_train.index}")
        # print(f"Test Index: {X_test.index}")
        # # print(f"X_train Info: {X_train.info()}")
        # # print(f"X_test Info: {X_test.info()}")

        seed = 99
        # if len(X_train) <= 100000 or X_train.shape[1] < 50:
        #     NUMS = 50
        #     FOLDS = 3
        # else:
        #     NUMS = 20
        #     FOLDS = 5

        NUMS = 50
        FOLDS = 3

        ## create Voting models
        estimators = []

        #######################################
        #### Cross Validation across Folds ####
        #######################################

        rmse_folds = []
        norm_rmse_folds = []
        forecast_df_folds = []
        
        
        NFOLDS = self.get_num_folds_from_cv(cv)

        # Here we are just taking 1 split to make it compatible with old version.
        cv = GapWalkForward(n_splits=1, gap_size=0, test_size=self.forecast_period)
        for fold_number, (train_idx, test_idx) in enumerate(cv.split(dfxs)):
            train = dfxs.iloc[train_idx]
            test = dfxs.iloc[test_idx]

            y_train = train[self.transformed_target]
            X_train = train[self.transformed_preds]
            
            y_test = test[self.transformed_target]
            X_test = test[self.transformed_preds]

            print(f"Using CV, Train Shape: {train.shape}")
            print(f"Using CV, Test Shape: {train.shape}")
            print(f"Using CV, Train Index: {test.index}")
            print(f"Using CV, Test Index: {test.index}")

          
        
        #### This section is for Time Series Models only ####
        if self.scoring == '':
            self.scoring = 'neg_mean_squared_error'
        tscv = TimeSeriesSplit(n_splits=FOLDS)
        tscv_new = GapWalkForward(n_splits=FOLDS, gap_size=0, test_size=self.forecast_period)

        for fold_number, (train_idx, test_idx) in enumerate(tscv.split(X_train)):
            ts_train = X_train.iloc[train_idx]
            ts_test = X_train.iloc[test_idx]

            print(f"FOLD {fold_number}")
            print(f"Using original Train Split Scheme, Train Shape: {ts_train.shape}")
            print(f"Using original Train Split Scheme, Test Shape: {ts_test.shape}")
            # print(f"Using original Train Split, Train Index: {ts_train.index}")
            # print(f"Using original Train Split, Test Index: {ts_test.index}")

        for fold_number, (train_idx, test_idx) in enumerate(tscv_new.split(X_train)):
            ts_train = X_train.iloc[train_idx]
            ts_test = X_train.iloc[test_idx]

            print(f"FOLD {fold_number}")
            print(f"Using CV Train Split, Train Shape: {ts_train.shape}")
            print(f"Using CV Train Split, Test Shape: {ts_test.shape}")
            # print(f"Using CV Train Split, Train Index: {ts_train.index}")
            # print(f"Using CV Train Split, Test Index: {ts_test.index}")


        self.scoring = 'neg_mean_squared_error'
        model5 = SVR(C=0.1, kernel='rbf', degree=2)
        results1 = cross_val_score(model5, X_train, y_train, cv=tscv, scoring=self.scoring)
        results1_alt = cross_val_score(model5, X_train, y_train, cv=tscv_new, scoring=self.scoring)
        print("SVR")
        print("Original results:")
        print(results1)
        print("New results")
        print(results1_alt)
        estimators.append(('SVR', model5, np.sqrt(abs(results1.mean()))))
        
        model6 = AdaBoostRegressor(
            base_estimator=DecisionTreeRegressor(
            min_samples_leaf=2, max_depth=1, random_state=seed),
            n_estimators=NUMS, random_state=seed
        )
        results2 = cross_val_score(model6, X_train, y_train, cv=tscv, scoring=self.scoring)
        results2_alt = cross_val_score(model6, X_train, y_train, cv=tscv_new, scoring=self.scoring)
        print("AdaBoost")
        print("Original results:")
        print(results2)
        print("New results")
        print(results2_alt)
        estimators.append(('Extra Trees', model6,np.sqrt(abs(results2.mean()))))
        
        model7 = LinearSVR(random_state=seed)
        results3 = cross_val_score(model7, X_train, y_train, cv=tscv, scoring=self.scoring)
        results3_alt = cross_val_score(model7, X_train, y_train, cv=tscv_new, scoring=self.scoring)
        print("Linear SVR")
        print("Original results:")
        print(results3)
        print("New results")
        print(results3_alt)
        estimators.append(('LinearSVR', model7, np.sqrt(abs(results3.mean()))))
        
        ## Create an ensemble model ####
        # estimators_list = [(tuples[0], tuples[1]) for tuples in estimators] # unused
        ensemble = BaggingRegressor(DecisionTreeRegressor(random_state=seed),
                                    n_estimators=NUMS, random_state=seed)
        results4 = cross_val_score(ensemble, X_train, y_train, cv=tscv, scoring=self.scoring)
        results4_alt = cross_val_score(ensemble, X_train, y_train, cv=tscv_new, scoring=self.scoring)
        print("Bagging")
        print("Original results:")
        print(results4)
        print("New results")
        print(results4_alt)
        estimators.append(('Bagging', ensemble, np.sqrt(abs(results4.mean()))))
        
        print('Running multiple models...')
        if self.verbose == 1:
            print('    Instance Based = %0.4f \n    Boosting = %0.4f\n    Linear Model = %0.4f \n    Bagging = %0.4f' %(
            np.sqrt(abs(results1.mean()))/y_train.std(), np.sqrt(abs(results2.mean()))/y_train.std(),
            np.sqrt(abs(results3.mean()))/y_train.std(), np.sqrt(abs(results4.mean()))/y_train.std()))
        
        besttype = sorted(estimators, key=lambda x: x[2], reverse=False)[0][0]
        print(f"Best Model: {besttype}")

        self.model = sorted(estimators, key=lambda x: x[2], reverse=False)[0][1]
        bestscore = sorted(estimators, key=lambda x: x[2], reverse=False)[0][2]/y_train.std()
        if self.verbose == 1:
            print('Best Model = %s with %0.2f Normalized RMSE score\n' % (besttype, bestscore))
        # print('Model Results:')
        

        ###############################################
        #### Refit the model on the entire dataset ####
        ###############################################

        # Refit Model on entire train dataset (earlier, we only trained the model on the individual splits)
        # Refit on the original dataframe minus the last `forecast_period` observations (train_orig_df)
        self.refit(ts_df=train_orig_df)

        # This is the new method without the leakage
        # Drop the y value
        test_orig_df_pred_only = test_orig_df.drop(self.original_target_col, axis=1, inplace=False)
        forecast = self.predict(X_exogen=test_orig_df_pred_only, simple=False)

        rmse, norm_rmse = print_dynamic_rmse(
            y_test.values,
            forecast['mean'],
            y_train.values
        )
        
        return self.model, forecast['mean'], rmse, norm_rmse

    def order_df(self, ts_df: pd.DataFrame) -> pd.DataFrame:
        """
        Given a dataframe (original), this will order the columns with 
        target as the first column and the predictors as the next set of columns
        The actual target being placed in the first column is not important.
        What is important is that the order of columns is maintained consistently
        throughout the lifecycle as this will also be used when predicting with
        test data (i.e. we need to prepend some of the train data to the test data
        before transforming the original time series dataframe into a supervised 
        learning problem.).
        """
        return ts_df[[self.original_target_col] + self.original_preds]


    def df_to_supervised(self, ts_df: pd.DataFrame) -> Tuple[pd.DataFrame, str, List[str]]:
        """
        :param ts_df: The time series dataframe that needs to be converted
        into a supervised learning problem.
        rtype: pd.DataFrame, str, List[str]
        """
        dfxs, transformed_target_name, transformed_pred_names = convert_timeseries_dataframe_to_supervised(
            ts_df[self.original_preds+[self.original_target_col]], 
            self.original_preds+[self.original_target_col],
            self.original_target_col,
            n_in=self.lags, n_out=0, dropT=False
        )
        # TODO: Call create_time_seties_features on dfxs along with the name of the time_series_index 
        # (index has to be converted to a column befoere passing to this)
        # This will retuen the same dataframe with 10 extra columns like day of week, weekend
        # Make your ML model  run like a charm.
        # Need to add new column names to the transformed pred_names.
        return dfxs, transformed_target_name, transformed_pred_names

    def refit(self, ts_df: pd.DataFrame) -> object:
        """
        :param ts_df The original dataframe. All transformations to a supervised learning
        problem should be taken care internally by this method.
        'target_col': and 'lags' do not need to be passed as was the case with the fit method.
        We will simply use the values that were stored during the training process.
        """
        
        self.check_model_built()
        
        dfxs, _, _  = self.df_to_supervised(ts_df)

        y_train = dfxs[self.transformed_target]
        X_train = dfxs[self.transformed_preds]

        self.model.fit(X_train, y_train)

        # Save last `self.lags` which will be used for predictions later
        self.df_train_prepend = ts_df[-self.lags:]

        return self
        

    def predict(
        self,
        X_exogen: Optional[pd.DataFrame]=None,
        forecast_period: Optional[int] = None,
        simple: bool = True) -> NDFrame:
        """
        Return the predictions
        :param: X_exogen The test dataframe in pretransformed format
        :param: forecast_period Not used this this case since for ML based models,
        X_egogen is a must, hence we can use the number of rows in X_egogen
        to get the forecast period. 
        """

        self.check_model_built()
        
        if X_exogen is None:
            warnings.warn(
                "You have not provided the exogenous variable in order to make the prediction. " + 
                "Machine Learing based models only support multivariate time series models. " +
                "Hence predictions will not be made.")
            return None

        # Extract the dynamic predicted and true values of our time series
        
        # Placebholder for forecasted results
        y_forecasted: List[float] = [] 

        ts_index = X_exogen.index
        # print(f"Datatime Index: {ts_index}")

        # STEP 1:       
        # self.df_prepend has the y column as well, but X_exogen does not. 
        # Need to add a dummy column to X_exogen before appending the 2 dataframes
        # However, Since we are going to depend on previous values of y column to make
        # future predictions, we can not just use all zeros for the y values
        # (especially for forrecasts beyond the 1st prediction). So we will 
        # make one prediction at a time and then use that prediction to make the next prediction.
        # That way, we get the most accurate prediction without leakage of informaton.
        
        # print (f"Columns before adding dummy: {X_exogen.columns}")
        X_exogen_with_dummy = X_exogen.copy(deep=True)

        # Just a check to make sure user is not passing the target column to predict function.
        if self.original_target_col in X_exogen_with_dummy.columns:
            warnings.warn("Your X_exogen dataframe contains the target column as well. This will be deleted for the predictions.")
            X_exogen_with_dummy.drop(self.original_target_col, axis=1, inplace=True)

        # Adding dummy value for target. 
        X_exogen_with_dummy[self.original_target_col] = np.zeros((X_exogen_with_dummy.shape[0],1))  

        # Make sure column order is correct when adding the dummy column
        X_exogen_with_dummy = self.order_df(X_exogen_with_dummy)
        # print (f"Columns after reordering: {X_exogen_with_dummy.columns}")

        # STEP 2:
        # Make prediction for each row. Then use the prediction for the next row.

        # TODO: Currently Frequency is missing in the data index (= None), so we can not shift the index
        ## When this is fixed in the AutoML module, we can shift and get the future index
        ## to be in a proper time series format.        
        # print("Train Prepend")
        # print(self.df_train_prepend)
        # index = self.df_train_prepend.index
        # print("Index Before")
        # print(index)
        # index = index.shift(X_exogen_with_dummy.shape[0])
        # print("Index After")
        # print(index)

        df_prepend = self.df_train_prepend.copy(deep=True)
        for i in np.arange(X_exogen_with_dummy.shape[0]):
            
            # Append the last n_lags of the data to the row of the X_egogen that is being preducted
            # Note that some of this will come from the last few observations of the training data
            # and the rest will come from the last few observations of the X_egogen data.
            # print(f"Prepend shape before adding test: {df_prepend.shape}")
            df_prepend = df_prepend.append(X_exogen_with_dummy.iloc[i])
            # print(f"Prepend shape after adding test: {df_prepend.shape}")
            # print("Prepend Dataframe")
            # print(df_prepend)

            # Convert the appended dataframe to supervised learning problem
            dfxs, _, _  = self.df_to_supervised(df_prepend)

            # Select only the predictors (transformed) from here
            X_test = dfxs[self.transformed_preds]    
            # print("X_test")
            # print(X_test)

            # Forecast
            y_forecasted_temp = self.model.predict(X_test)  # Numpy array
            # print(y_forecasted_temp) 
            y_forecasted.append(y_forecasted_temp[0])

            # Append the predicted value for use in next prediction
            df_prepend.iloc[-1][self.original_target_col] = y_forecasted_temp[0]

            # Remove 1st entry as it is not needed for next round
            df_prepend = df_prepend[1:]

            # print("df_prepend end of loop")
            # print(df_prepend)

        # y_forecasted = np.array(y_forecasted)
        res_frame = pd.DataFrame({'mean': y_forecasted})
        res_frame.index = ts_index
        res_frame['mean_se'] = np.nan
        res_frame['mean_ci_lower'] = np.nan
        res_frame['mean_ci_upper'] = np.nan

        if simple:
            return res_frame['mean']
        else:
            return res_frame
