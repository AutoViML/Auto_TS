from typing import List, Optional
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
       

    #def fit(self, X, Y, modeltype='Regression', scoring='', verbose=0):
    def fit(self, ts_df: pd.DataFrame, target_col: str, lags: int = 0):
        """
        Build a Time Series Model using Machine Learning models.
        Quickly builds and runs multiple models for a clean data set (only numerics).
        """

        self.original_target_col = target_col
        self.lags = lags
        self.original_preds = [x for x in list(ts_df) if x not in [self.original_target_col]]

        dfxs, target, preds = convert_timeseries_dataframe_to_supervised(
            ts_df[self.original_preds+[self.original_target_col]], 
            self.original_preds+[self.original_target_col],
            self.original_target_col,
            n_in=self.lags, n_out=0, dropT=False
        )
        
        train = dfxs[:-self.forecast_period]
        test = dfxs[-self.forecast_period:]

        Y = train[target]
        X = train[preds]
        
        y_test = test[target]
        X_test = test[preds]
        
        seed = 99
        if len(X) <= 100000 or X.shape[1] < 50:
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
        results1 = cross_val_score(model5, X, Y, cv=tscv, scoring=self.scoring)
        estimators.append(('SVR', model5, np.sqrt(abs(results1.mean()))))
        model6 = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(
                                min_samples_leaf=2, max_depth=1, random_state=seed),
                                n_estimators=NUMS, random_state=seed)
        results2 = cross_val_score(model6, X, Y, cv=tscv, scoring=self.scoring)
        estimators.append(('Extra Trees', model6,np.sqrt(abs(results2.mean()))))
        model7 = LinearSVR(random_state=seed)
        results3 = cross_val_score(model7, X, Y, cv=tscv, scoring=self.scoring)
        estimators.append(('LinearSVR', model7, np.sqrt(abs(results3.mean()))))
        ## Create an ensemble model ####
        # estimators_list = [(tuples[0], tuples[1]) for tuples in estimators] # unused
        ensemble = BaggingRegressor(DecisionTreeRegressor(random_state=seed),
                                    n_estimators=NUMS, random_state=seed)
        results4 = cross_val_score(ensemble, X, Y, cv=tscv, scoring=self.scoring)
        estimators.append(('Bagging', ensemble, np.sqrt(abs(results4.mean()))))
        print('Running multiple models...')
        if self.verbose == 1:
            print('    Instance Based = %0.4f \n    Boosting = %0.4f\n    Linear Model = %0.4f \n    Bagging = %0.4f' %(
            np.sqrt(abs(results1.mean()))/Y.std(), np.sqrt(abs(results2.mean()))/Y.std(),
            np.sqrt(abs(results3.mean()))/Y.std(), np.sqrt(abs(results4.mean()))/Y.std()))
        besttype = sorted(estimators, key=lambda x: x[2], reverse=False)[0][0]
        self.model = sorted(estimators, key=lambda x: x[2], reverse=False)[0][1]
        bestscore = sorted(estimators, key=lambda x: x[2], reverse=False)[0][2]/Y.std()
        if self.verbose == 1:
            print('Best Model = %s with %0.2f Normalized RMSE score\n' % (besttype, bestscore))
        # print('Model Results:')
        
        # Refit Model on entire train dataset (earlier, we only did on splits)
        self.model.fit(X, Y)
        forecast = self.model.predict(X_test)
        rmse, norm_rmse = print_dynamic_rmse(
            y_test.values,
            forecast,
            Y.values
        )
        
        # return self.model, bestscore, besttype
        return self.model, forecast, rmse, norm_rmse


    def predict(self, forecast_period: Optional[int] = None):
        """
        Return the predictions
        """
        # Extract the dynamic predicted and true values of our time series
        if forecast_period is None:
            # use the forecast period used during training
            y_forecasted = self.model.forecast(self.forecast_period)
        else:
            # use the forecast period provided by the user
            y_forecasted = self.model.forecast(forecast_period)
        return y_forecasted
