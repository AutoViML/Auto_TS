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
        
        # Convert to supervised learning problem
        dfxs, self.transformed_target, self.transformed_preds = self.df_to_supervised(ts_df)
                
        ## create Voting models
        estimators = []

        #######################################
        #### Cross Validation across Folds ####
        #######################################

        rmse_folds = []
        norm_rmse_folds = []
        forecast_df_folds = []  # TODO: See if this can be retreived somehow
        
        NFOLDS = self.get_num_folds_from_cv(cv)

        seed = 99
        
        X_train = dfxs[self.transformed_preds]
        y_train = dfxs[self.transformed_target]

        # Decide NUM_ESTIMATORS for trees
        if len(X_train) <= 100000 or X_train.shape[1] < 50:
            NUMS = 50
        else:
            NUMS = 20
       
        if self.scoring == '':
            self.scoring = 'neg_root_mean_squared_error'
        elif self.scoring == 'rmse':
            self.scoring = 'neg_root_mean_squared_error'
        
        ts_cv = GapWalkForward(n_splits=NFOLDS, gap_size=0, test_size=self.forecast_period)

        if self.verbose >= 1:
            print('Running multiple models...')
        
        model5 = SVR(C=0.1, kernel='rbf', degree=2)
        results1 = cross_val_score(model5, X_train, y_train, cv=ts_cv, scoring=self.scoring)
        estimators.append(('SVR', model5, abs(results1.mean()), abs(results1)  ))
        
        model6 = AdaBoostRegressor(
            base_estimator=DecisionTreeRegressor(
            min_samples_leaf=2, max_depth=1, random_state=seed),
            n_estimators=NUMS, random_state=seed
        )
        results2 = cross_val_score(model6, X_train, y_train, cv=ts_cv, scoring=self.scoring)
        estimators.append(('Extra Trees', model6, abs(results2.mean()), abs(results2)  ))
        
        model7 = LinearSVR(random_state=seed)
        results3 = cross_val_score(model7, X_train, y_train, cv=ts_cv, scoring=self.scoring)
        estimators.append(('LinearSVR', model7, abs(results3.mean()), abs(results3)  ))
        
        ## Create an ensemble model ####
        ensemble = BaggingRegressor(DecisionTreeRegressor(random_state=seed),
                                    n_estimators=NUMS, random_state=seed)
        results4 = cross_val_score(ensemble, X_train, y_train, cv=ts_cv, scoring=self.scoring)
        estimators.append(('Bagging', ensemble, abs(results4.mean()), abs(results4)  )) 
        
        if self.verbose == 1:
            print('    Instance Based = %0.4f \n    Boosting = %0.4f\n    Linear Model = %0.4f \n    Bagging = %0.4f' %(
            abs(results1.mean())/y_train.std(), abs(results2.mean())/y_train.std(),
            abs(results3.mean())/y_train.std(), abs(results4.mean())/y_train.std()))
        
        besttype = sorted(estimators, key=lambda x: x[2], reverse=False)[0][0]
        # print(f"Best Model: {besttype}")

        self.model = sorted(estimators, key=lambda x: x[2], reverse=False)[0][1]
        bestscore = sorted(estimators, key=lambda x: x[2], reverse=False)[0][2]/y_train.std()
        rmse_folds = sorted(estimators, key=lambda x: x[2], reverse=False)[0][3]
        norm_rmse_folds = rmse_folds/y_train.values.std()  # Same as what was there in print_dynamic_rmse()
        
        if self.verbose == 1:
            print('Best Model = %s with %0.2f Normalized RMSE score\n' % (besttype, bestscore))
        

        ###############################################
        #### Refit the model on the entire dataset ####
        ###############################################

        # Refit Model on entire train dataset (earlier, we only trained the model on the individual splits)
        self.refit(ts_df=ts_df)

        # # This is the new method without the leakage
        # # Drop the y value
        # test_orig_df_pred_only = test_orig_df.drop(self.original_target_col, axis=1, inplace=False)
        # forecast = self.predict(X_exogen=test_orig_df_pred_only, simple=False)

        # rmse, norm_rmse = print_dynamic_rmse(
        #     y_test.values,
        #     forecast['mean'],
        #     y_train.values
        # )

        # print(f"RMSE Folds: {rmse_folds}")
        # print(f"Norm RMSE Folds: {norm_rmse_folds}")
        
        # return self.model, forecast['mean'], rmse, norm_rmse
        return self.model, forecast_df_folds, rmse_folds, norm_rmse_folds

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
