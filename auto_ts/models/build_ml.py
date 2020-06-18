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
from ..utils import print_static_rmse, print_dynamic_rmse
import pdb

class BuildML():
    def __init__(self, scoring='', verbose=0):
        """
        Automatically build a ML Model
        """
        self.scoring = scoring
        self.verbose = verbose
        self.model = None
       

    #def fit(self, X, Y, modeltype='Regression', scoring='', verbose=0):
    def fit(self, X, Y):
        """
        Build a Time Series Model using Machine Learning models.
        Quickly builds and runs multiple models for a clean data set (only numerics).
        """

        seed = 99
        if len(X) <= 100000 or X.shape[1] < 50:
            NUMS = 50
            FOLDS = 3
        else:
            NUMS = 20
            FOLDS = 5

        ## create Voting models
        estimators = []
        # if modeltype.lower() == 'regression':
        #     if self.scoring == '':
        #         self.scoring = 'neg_mean_squared_error'
        #     scv = ShuffleSplit(n_splits=FOLDS, random_state=seed)
        #     model5 = LinearRegression()
        #     results1 = cross_val_score(model5, X, Y, cv=scv, scoring=self.scoring)
        #     estimators.append(('Linear Model', model5, np.sqrt(abs(results1.mean()))))
        #     model6 = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(
        #                             min_samples_leaf=2, max_depth=1, random_state=seed),
        #                             n_estimators=NUMS, random_state=seed)
        #     results2 = cross_val_score(model6, X, Y, cv=scv, scoring=self.scoring)
        #     estimators.append(('Boosting', model6, np.sqrt(abs(results2.mean()))))
        #     model7 = RidgeCV(alphas=np.logspace(-10, -1, 50), cv=scv)
        #     results3 = cross_val_score(model7, X, Y, cv=scv, scoring=self.scoring)
        #     estimators.append(('Linear Regularization', model7, np.sqrt(abs(results3.mean()))))
        #     ## Create an ensemble model ####
        #     # estimators_list = [(tuples[0], tuples[1]) for tuples in estimators] # unused
        #     ensemble = BaggingRegressor(DecisionTreeRegressor(random_state=seed),
        #                                 n_estimators=NUMS, random_state=seed)
        #     results4 = cross_val_score(ensemble, X, Y, cv=scv, scoring=self.scoring)
        #     estimators.append(('Bagging', ensemble, np.sqrt(abs(results4.mean()))))
        #     if self.verbose == 1:
        #         print('\nLinear Model = %0.4f \nBoosting = %0.4f\nRegularization = %0.4f \nBagging = %0.4f' %(
        #         np.sqrt(abs(results1.mean()))/Y.std(), np.sqrt(abs(results2.mean()))/Y.std(),
        #         np.sqrt(abs(results3.mean()))/Y.std(), np.sqrt(abs(results4.mean()))/Y.std()))
        #     besttype = sorted(estimators, key=lambda x: x[2], reverse=False)[0][0]
        #     self.model = sorted(estimators, key=lambda x: x[2], reverse=False)[0][1]
        #     bestscore = sorted(estimators, key=lambda x: x[2], reverse=False)[0][2]/Y.std()
        #     if self.verbose == 1:
        #         print('    Best Model = %s with %0.2f Normalized RMSE score\n' %(besttype,bestscore))
        

        #elif modeltype.lower() == 'timeseries' or modeltype.lower() =='time series' or modeltype.lower() == 'time_series':
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
        
        # else:
        #     if self.scoring == '':
        #         self.scoring = 'f1'
        #     scv = StratifiedShuffleSplit(n_splits=FOLDS, random_state=seed)
        #     model5 = LogisticRegression(random_state=seed)
        #     results1 = cross_val_score(model5, X, Y, cv=scv, scoring=self.scoring)
        #     estimators.append(('Logistic Regression', model5, abs(results1.mean())))
        #     model6 = LinearDiscriminantAnalysis()
        #     results2 = cross_val_score(model6, X, Y, cv=scv, scoring=self.scoring)
        #     estimators.append(('Linear Discriminant', model6, abs(results2.mean())))
        #     model7 = ExtraTreesClassifier(n_estimators=NUMS, min_samples_leaf=2, random_state=seed)
        #     results3 = cross_val_score(model7, X, Y, cv=scv, scoring=self.scoring)
        #     estimators.append(('Bagging', model7, abs(results3.mean())))
        #     ## Create an ensemble model ####
        #     # estimators_list = [(tuples[0], tuples[1]) for tuples in estimators] # unused
        #     ensemble = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(
        #                                 random_state=seed, max_depth=1, min_samples_leaf=2),
        #                                 n_estimators=NUMS, random_state=seed)
        #     results4 = cross_val_score(ensemble, X, Y, cv=scv, scoring=self.scoring)
        #     estimators.append(('Boosting', ensemble, abs(results4.mean())))
        #     if self.verbose == 1:
        #         print('\nLogistic Regression = %0.4f \nLinear Discriminant = %0.4f \nBagging = %0.4f \nBoosting = %0.4f' %
        #             (abs(results1.mean()), abs(results2.mean()), abs(results3.mean()), abs(results4.mean())))
        #     besttype = sorted(estimators, key=lambda x: x[2], reverse=True)[0][0]
        #     self.model = sorted(estimators, key=lambda x: x[2], reverse=True)[0][1]
        #     bestscore = sorted(estimators, key=lambda x: x[2], reverse=True)[0][2]
        #     if self.verbose == 1:
        #         print('    Best Model = %s with %0.2f %s score\n' % (besttype, bestscore, self.scoring))
        
        return self.model, bestscore, besttype


    # def predict(self, forecast_period: Optional[int] = None):
    #     """
    #     Return the predictions
    #     """
    #     # Extract the dynamic predicted and true values of our time series
    #     if forecast_period is None:
    #         # use the forecast period used during training
    #         y_forecasted = self.model.forecast(self.forecast_period)
    #     else:
    #         # use the forecast period provided by the user
    #         y_forecasted = self.model.forecast(forecast_period)
    #     return y_forecasted
