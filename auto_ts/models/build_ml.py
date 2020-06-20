import warnings
from typing import List, Optional, Tuple
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
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
# imported specialized tree models from scikit-garden
# from skgarden import RandomForestQuantileRegressor
# helper functions
from ..utils import print_static_rmse, print_dynamic_rmse, convert_timeseries_dataframe_to_supervised
import pdb

class BuildML():
    def __init__(self, scoring='', forecast_period=2, verbose=0):
        """
        Automatically build a ML Model
        """
        self.scoring = scoring
        self.forecast_period = forecast_period
        self.verbose = verbose
        self.model = None

        # Specific to ML model
        # These are needed so that during prediction later, the data can be transformed correctly
        self.original_target_col: str = ""
        self.original_preds: List[str] = []
        self.lags: int = 0

        self.transformed_target: str = ""
        self.transformed_preds: List[str] = []

        # This saves the last `self.lags` of the original train dataframe
        # This is needed during predictions to transform the X_test
        # to a supervised learning problem.
        self.df_train_prepend: pd.DataFrame = pd.DataFrame()
       

    #def fit(self, X, Y, modeltype='Regression', scoring='', verbose=0):
    def fit(self, ts_df: pd.DataFrame, target_col: str, lags: int = 0):
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
        
        train = dfxs[:-self.forecast_period]
        test = dfxs[-self.forecast_period:]

        y_train = train[self.transformed_target]
        X_train = train[self.transformed_preds]
        
        y_test = test[self.transformed_target]
        X_test = test[self.transformed_preds]
        
        # print("ML Diagnostics:")
        # print(f"Lags: {self.lags}")
        # print(f"Original Shape: {ts_df.shape}")
        # print(f"Transformed Shape: {dfxs.shape}")
        # print(f"Forecast Period: {self.forecast_period}")
        # print(f"Train Shape: {X_train.shape}")
        # print(f"Test Shape: {X_test.shape}")

        seed = 99
        if len(X_train) <= 100000 or X_train.shape[1] < 50:
            NUMS = 50
            FOLDS = 3
        else:
            NUMS = 20
            FOLDS = 5

        ## create Voting models
        estimators = []
            
        
        #### This section is for Time Series Models only ####
        if self.scoring == '':
            self.scoring = 'neg_mean_squared_error'
        tscv = TimeSeriesSplit(n_splits=FOLDS)
        self.scoring = 'neg_mean_squared_error'
        model5 = SVR(C=0.1, kernel='rbf', degree=2)
        results1 = cross_val_score(model5, X_train, y_train, cv=tscv, scoring=self.scoring)
        estimators.append(('SVR', model5, np.sqrt(abs(results1.mean()))))
        model6 = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(
                                min_samples_leaf=2, max_depth=1, random_state=seed),
                                n_estimators=NUMS, random_state=seed)
        results2 = cross_val_score(model6, X_train, y_train, cv=tscv, scoring=self.scoring)
        estimators.append(('Extra Trees', model6,np.sqrt(abs(results2.mean()))))
        model7 = LinearSVR(random_state=seed)
        results3 = cross_val_score(model7, X_train, y_train, cv=tscv, scoring=self.scoring)
        estimators.append(('LinearSVR', model7, np.sqrt(abs(results3.mean()))))
        ## Create an ensemble model ####
        # estimators_list = [(tuples[0], tuples[1]) for tuples in estimators] # unused
        ensemble = BaggingRegressor(DecisionTreeRegressor(random_state=seed),
                                    n_estimators=NUMS, random_state=seed)
        results4 = cross_val_score(ensemble, X_train, y_train, cv=tscv, scoring=self.scoring)
        estimators.append(('Bagging', ensemble, np.sqrt(abs(results4.mean()))))
        print('Running multiple models...')
        if self.verbose == 1:
            print('    Instance Based = %0.4f \n    Boosting = %0.4f\n    Linear Model = %0.4f \n    Bagging = %0.4f' %(
            np.sqrt(abs(results1.mean()))/y_train.std(), np.sqrt(abs(results2.mean()))/y_train.std(),
            np.sqrt(abs(results3.mean()))/y_train.std(), np.sqrt(abs(results4.mean()))/y_train.std()))
        besttype = sorted(estimators, key=lambda x: x[2], reverse=False)[0][0]
        self.model = sorted(estimators, key=lambda x: x[2], reverse=False)[0][1]
        bestscore = sorted(estimators, key=lambda x: x[2], reverse=False)[0][2]/y_train.std()
        if self.verbose == 1:
            print('Best Model = %s with %0.2f Normalized RMSE score\n' % (besttype, bestscore))
        # print('Model Results:')
        
        # Refit Model on entire train dataset (earlier, we only trained the model on the individual splits)
        # Refit on the original dataframe minus the last `forecast_period` observations (train_orig_df)
        self.refit(ts_df=train_orig_df)

        # This is the old method that had leakage (X_test has known values of predicted y values)    
        # forecast = self.model.predict(X_test)
        # print("Forecasts (old method)")
        # print(forecast)

        # This is the new method without the leakage
        # Drop the y value
        test_orig_df_pred_only = test_orig_df.drop(self.original_target_col, axis=1, inplace=False)
        forecast = self.predict(test_orig_df_pred_only)
        # print("Forecasts (new)")
        # print(forecast)

        rmse, norm_rmse = print_dynamic_rmse(
            y_test.values,
            forecast,
            y_train.values
        )
        
        return self.model, forecast, rmse, norm_rmse

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
        return dfxs, transformed_target_name, transformed_pred_names

    def refit(self, ts_df: pd.DataFrame):
        """
        :param ts_df The original dataframe. All transformations to a supervised learning
        problem should be taken care internally by this method.
        'target_col': and 'lags' do not need to be passed as was the case with the fit method.
        We will simply use the values that were stored during the training process.
        """
        # TODO: Add for ML model
        # Reuse the names obtained during original training
        #dfxs, self.transformed_target, self.transformed_preds = self.df_to_supervised(ts_df)
        dfxs, _, _  = self.df_to_supervised(ts_df)

        y_train = dfxs[self.transformed_target]
        X_train = dfxs[self.transformed_preds]

        self.model.fit(X_train, y_train)

        # Save last `self.lags` which will be used for predictions later
        self.df_train_prepend = ts_df[-self.lags:]
        

    def predict(self, X_exogen: pd.DataFrame, forecast_period: Optional[int] = None):
        """
        Return the predictions
        :param: X_exogen The test dataframe in pretransformed format
        :param: forecast_period Not used this this case since for ML based models,
        X_egogen is a must, hence we can use the number of rows in X_egogen
        to get the forecast period. 
        """

        # Extract the dynamic predicted and true values of our time series
        
        # Placebholder for forecasted results
        y_forecasted: List[float] = [] 

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

        y_forecasted = np.array(y_forecasted)

        return y_forecasted
