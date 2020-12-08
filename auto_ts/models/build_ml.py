import warnings
import copy
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
        dfxs, self.transformed_target, self.transformed_preds = self.df_to_supervised(
            ts_df=ts_df, drop_zero_var = True)


        print("Fitting ML model")
        # print(f"Transformed DataFrame:")
        # print(dfxs.info())
        # print(f"Transformed Target: {self.transformed_target}")
        # print(f"Transformed Predictors: {self.transformed_preds}")


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


    def df_to_supervised(
        self,
        ts_df: pd.DataFrame,
        drop_zero_var: bool = False) -> Tuple[pd.DataFrame, str, List[str]]:
        """
        :param ts_df: The time series dataframe that needs to be converted
        into a supervised learning problem.
        rtype: pd.DataFrame, str, List[str]
        """
        dfxs, transformed_target_name, _ = convert_timeseries_dataframe_to_supervised(
            ts_df[self.original_preds+[self.original_target_col]],
            self.original_preds+[self.original_target_col],
            self.original_target_col,
            n_in=self.lags, n_out=0, dropT=False
        )

        # Append the time series features (derived from the time series index)
        # None ts_column will use the index
        dfxs = create_time_series_features(dtf=dfxs, ts_column=None, drop_zero_var=drop_zero_var)

        # Overwrite with new ones
        # transformed_pred_names = [x for x in list(dfxs) if x not in [self.transformed_target]]
        transformed_pred_names = [x for x in list(dfxs) if x not in [transformed_target_name]]

        return dfxs, transformed_target_name, transformed_pred_names

    def refit(self, ts_df: pd.DataFrame) -> object:
        """
        :param ts_df The original dataframe. All transformations to a supervised learning
        problem should be taken care internally by this method.
        'target_col': and 'lags' do not need to be passed as was the case with the fit method.
        We will simply use the values that were stored during the training process.
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
        simple: bool = True) -> NDFrame:
        """
        Return the predictions
        :param: testdata The test dataframe in pretransformed format
        :param: forecast_period Not used this this case since for ML based models,
        X_egogen is a must, hence we can use the number of rows in X_egogen
        to get the forecast period.
        """

        self.check_model_built()

        if testdata is None:
            warnings.warn(
                "You have not provided the exogenous variable in order to make the prediction. " +
                "Machine Learing based models only support multivariate time series models. " +
                "Hence predictions will not be made.")
            return None

        # Extract the dynamic predicted and true values of our time series

        # Placebholder for forecasted results
        y_forecasted: List[float] = []

        ts_index = testdata.index
        # print(f"Datatime Index: {ts_index}")

        # STEP 1:
        # self.df_prepend has the y column as well, but testdata does not.
        # Need to add a dummy column to testdata before appending the 2 dataframes
        # However, Since we are going to depend on previous values of y column to make
        # future predictions, we can not just use all zeros for the y values
        # (especially for forrecasts beyond the 1st prediction). So we will
        # make one prediction at a time and then use that prediction to make the next prediction.
        # That way, we get the most accurate prediction without leakage of informaton.

        # print (f"Columns before adding dummy: {testdata.columns}")
        testdata_with_dummy = testdata.copy(deep=True)

        # Just a check to make sure user is not passing the target column to predict function.
        if self.original_target_col in testdata_with_dummy.columns:
            warnings.warn("Your testdata dataframe contains the target column as well. This will be deleted for the predictions.")
            testdata_with_dummy.drop(self.original_target_col, axis=1, inplace=True)

        # Adding dummy value for target.
        testdata_with_dummy[self.original_target_col] = np.zeros((testdata_with_dummy.shape[0],1))

        # Make sure column order is correct when adding the dummy column
        testdata_with_dummy = self.order_df(testdata_with_dummy)
        # print (f"Columns after reordering: {testdata_with_dummy.columns}")

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
        # index = index.shift(testdata_with_dummy.shape[0])
        # print("Index After")
        # print(index)

        df_prepend = self.df_train_prepend.copy(deep=True)
        for i in np.arange(testdata_with_dummy.shape[0]):

            # Append the last n_lags of the data to the row of the X_egogen that is being preducted
            # Note that some of this will come from the last few observations of the training data
            # and the rest will come from the last few observations of the X_egogen data.
            # print(f"Prepend shape before adding test: {df_prepend.shape}")
            df_prepend = df_prepend.append(testdata_with_dummy.iloc[i])
            # print(f"Prepend shape after adding test: {df_prepend.shape}")
            # print("Prepend Dataframe")
            # print(df_prepend)

            # Convert the appended dataframe to supervised learning problem
            dfxs, _, _  = self.df_to_supervised(ts_df=df_prepend, drop_zero_var=False)

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


def create_time_series_features(dtf, ts_column: Optional[str]=None, drop_zero_var: bool = False):
    """
    This creates between 8 and 10 date time features for each date variable.
    The number of features depends on whether it is just a year variable
    or a year+month+day and whether it has hours and mins+secs. So this can
    create all these features using just the date time column that you send in.
    It returns the entire dataframe with added variables as output.
    """
    dtf = copy.deepcopy(dtf)

    try:
        # ts_column = None assumes that that index is the time series index
        reset_index = False
        if ts_column is None:
            reset_index = True
            ts_column = dtf.index.name
            dtf.reset_index(inplace=True)

        ### In some extreme cases, date time vars are not processed yet and hence we must fill missing values here!
        if dtf[ts_column].isnull().sum() > 0:
            # missing_flag = True
            new_missing_col = ts_column + '_Missing_Flag'
            dtf[new_missing_col] = 0
            dtf.loc[dtf[ts_column].isnull(),new_missing_col]=1
            dtf[ts_column] = dtf[ts_column].fillna(method='ffill')

        if dtf[ts_column].dtype == float:
            dtf[ts_column] = dtf[ts_column].astype(int)

        ### if we have already found that it was a date time var, then leave it as it is. Thats good enough!

        items = dtf[ts_column].apply(str).apply(len).values
        #### In some extreme cases,
        if all(items[0] == item for item in items):
            if items[0] == 4:
                ### If it is just a year variable alone, you should leave it as just a year!
                dtf[ts_column] = pd.to_datetime(dtf[ts_column],format='%Y')
            else:
                ### if it is not a year alone, then convert it into a date time variable
                dtf[ts_column] = pd.to_datetime(dtf[ts_column], infer_datetime_format=True)
        else:
            dtf[ts_column] = pd.to_datetime(dtf[ts_column], infer_datetime_format=True)

        dtf = create_ts_features(df=dtf, tscol=ts_column, drop_zero_var=drop_zero_var, return_original=True)

        # If you had reset the index earlier, set it back before returning
        # to  make it consistent with the dataframe that was sent as input
        if reset_index:
            dtf.set_index(ts_column, inplace=True)

    except Exception as e:
        print(e)
        print('Error in Processing %s column for date time features. Continuing...' %ts_column)

    return dtf


def create_ts_features(
    df,
    tscol,
    drop_zero_var: bool = True,
    return_original: bool = True) -> pd.DataFrame:
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
        df[tscol+'_minute'] = df[tscol].dt.minute.astype(int)
        dt_adds.append(tscol+'_hour')
        dt_adds.append(tscol+'_minute')
    except:
        print('    Error in creating hour-second derived features. Continuing...')
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
