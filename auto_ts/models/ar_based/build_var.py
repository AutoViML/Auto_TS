"""Module to build a VAR model
"""
import pdb
from typing import Optional
import warnings
import itertools
import operator
import copy

import numpy as np  # type: ignore
import pandas as pd # type: ignore
from pandas.core.generic import NDFrame # type:ignore
import dask
import dask.dataframe as dd

import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
sns.set(style="white", color_codes=True)

from statsmodels.tsa.statespace.varmax import VARMAX # type: ignore

#from tscv import GapWalkForward # type: ignore
from sklearn.model_selection import TimeSeriesSplit

# helper functions
from ...utils import print_dynamic_rmse
from ...models.ar_based.param_finder import find_lowest_pq
from ..build_base import BuildBase


class BuildVAR(BuildBase):
    """Class to build a VAR model
    """
    def __init__(self, scoring, forecast_period=2, p_max=3, q_max=3, verbose=0):
        """
        Automatically build a VAR Model

        Since it automatically builds a VAR model, you need to give it a Criteria (scoring) to optimize
        on. You can give it any of the following metrics as scoring options:
            AIC, BIC, Deviance, Log-likelihood.
        You can give the highest order values for p and q. Default is set to 3 for both.
        """
        super().__init__(
            scoring=scoring,
            forecast_period=forecast_period,
            verbose=verbose
        )
        self.p_max = p_max
        self.q_max = q_max
        self.best_p = None
        self.best_d = None
        self.best_q = None

    # def fit(self, ts_df):
    def fit(self, ts_df: pd.DataFrame, target_col: str, cv: Optional[int] = None) -> object:
        """
         This builds a VAR model given a multivariate time series data frame with time as the Index.

        :param ts_df The time series data to be used for fitting the model. Note that the input can be
        a data frame with one column or multiple cols or a multivariate array. However, the first column
        must be the target variable. You must include only Time Series data in it. DO NOT include
        "Non-Stationary" or "Trendy" data. Make sure your Time Series is "Stationary" before you send
        it in!! If not, this will give spurious results.
        :type ts_df pd.DataFrame

        :param target_col The column name of the target time series that needs to be modeled.
        All other columns will be considered as exogenous variables (if applicable to method)
        :type target_col str

        :param cv: Number of folds to use for cross validation.
        Number of observations in the Validation set for each fold = forecast period
        If None, a single fold is used
        :type cv Optional[int]

        :rtype object
        """
        self.original_target_col = target_col
        self.original_preds = [x for x in list(ts_df) if x not in [self.original_target_col]]

        ts_df = ts_df[[self.original_target_col] + self.original_preds]

        #######################################
        #### Cross Validation across Folds ####
        #######################################

        rmse_folds = []
        norm_rmse_folds = []
        forecast_df_folds = []
        norm_rmse_folds2 = []

        ### Creating a new way to skip cross validation when trying to run auto-ts multiple times. ###
        if not cv:
            cv_in = 0
        else:
            cv_in = copy.deepcopy(cv)
        NFOLDS = self.get_num_folds_from_cv(cv)
        #cv = GapWalkForward(n_splits=NFOLDS, gap_size=0, test_size=self.forecast_period)
        #cv = TimeSeriesSplit(n_splits=NFOLDS, test_size=self.forecast_period) ### sklearn version 0.0.24
        max_trainsize = len(ts_df) - self.forecast_period
        try:
            cv = TimeSeriesSplit(n_splits=NFOLDS, test_size=self.forecast_period) ### this works only sklearn v 0.0.24]
        except:
            cv = TimeSeriesSplit(n_splits=NFOLDS, max_train_size = max_trainsize)

        if type(ts_df) == dask.dataframe.core.DataFrame:
            ts_df = dft.head(len(ts_df)) ### this converts dask into a pandas dataframe

        if  cv_in == 0:
            print('Skipping cross validation steps since cross_validation = %s' %cv_in)
            self.find_best_parameters(data = ts_df)
            y_train = ts_df.iloc[:, [0, self.best_d]]
            bestmodel = self.get_best_model(y_train)
            self.model = bestmodel.fit(disp=False)
        else:
            for fold_number, (train_index, test_index) in enumerate(cv.split(ts_df)):
                dftx = ts_df.head(len(train_index)+len(test_index))
                ts_train = dftx.head(len(train_index)) ## now train will be the first segment of dftx
                ts_test = dftx.tail(len(test_index)) ### now test will be right after train in dftx

                print(f"\nFold Number: {fold_number+1} --> Train Shape: {ts_train.shape[0]} Test Shape: {ts_test.shape[0]}")
                self.find_best_parameters(data = ts_train)

                #########################################
                #### Define the model with fold data ####
                #########################################
                y_train = ts_train.iloc[:, [0, self.best_d]]
                bestmodel = self.get_best_model(y_train)

                ######################################
                #### Fit the model with fold data ####
                ######################################

                if self.verbose >= 1:
                    print(f'Fitting best VAR model on Fold: {fold_number+1}')
                try:
                    self.model = bestmodel.fit(disp=False)
                except Exception as e:
                    print(e)
                    print(f'Error: VAR Fit on Fold: {fold_number+1} unsuccessful.')
                    return bestmodel, None, np.inf, np.inf

                forecast_df = self.predict(ts_test.shape[0],simple=False)
                forecast_df_folds.append(forecast_df['yhat'].values)

                rmse, norm_rmse = print_dynamic_rmse(ts_test.iloc[:, 0].values, forecast_df['yhat'].values,
                                            ts_train.iloc[:, 0].values)
                rmse_folds.append(rmse)
                norm_rmse_folds.append(norm_rmse)

            norm_rmse_folds2 = rmse_folds/ts_df[self.original_target_col].values.std()  # Same as what was there in print_dynamic_rmse()
            self.model.plot_diagnostics(figsize=(16, 12))
            axis = self.model.impulse_responses(12, orthogonalized=True).plot(figsize=(12, 4))
            axis.set(xlabel='Time Steps', title='VAR model Impulse Response Functions')

        ###############################################
        #### Refit the model on the entire dataset ####
        ###############################################
        y_train = ts_df.iloc[:, [0, self.best_d]]
        self.refit(ts_df=y_train)

        # return self.model, forecast_df_folds, rmse_folds, norm_rmse_folds
        return self.model, forecast_df_folds, rmse_folds, norm_rmse_folds2

    def predict(
            self,
            testdata: Optional[pd.DataFrame] = None,
            forecast_period: Optional[int] = None,
            simple: bool = True
        ) -> NDFrame:
        """
        Return the predictions
        """

        if testdata is not None:
            warnings.warn(
                "You have passed exogenous variables to make predictions for a VAR model. " +
                "VAR model will predict all exogenous variables automatically, " +
                "hence your passed values will not be used."
            )
            if isinstance(testdata, pd.DataFrame) or isinstance(testdata, pd.Series):
                if len(testdata) != self.forecast_period:
                    self.forecast_period = testdata.shape[0]
            elif isinstance(testdata, int):
                self.forecast_period = testdata

            forecast_period = self.forecast_period

        # Extract the dynamic predicted and true values of our time series
        if forecast_period is None:
            # use the forecast period used during training
            forecast_period = self.forecast_period

        # y_forecasted = self.model.forecast(forecast_period)

        res = self.model.get_forecast(forecast_period)
        res_frame = res.summary_frame()

        res_frame.rename(columns={'mean':'yhat'},inplace=True)

        if simple:
            res_frame = res_frame['yhat']
            res_frame = res_frame.squeeze() # Convert to a pandas series object
        else:
            # Pass as is
            pass

        return res_frame


    def find_best_parameters(self, data: pd.DataFrame):
        """
        Given a dataset, finds the best parameters using the settings in the class
        """
        #### dmax here means the column number of the data frame: it serves as a placeholder for columns
        dmax = data.shape[1]
        ###############################################################################################
        cols = data.columns.tolist()
        # TODO: #14 Make sure that we have a way to not rely on column order to determine the target
        # It is assumed that the first column of the dataframe is the target variable ####
        ### make sure that is the case before doing this program ####################
        i = 1
        results_dict = {}
        
        for d_val in range(1, dmax):
            # Takes the target column and one other endogenous column at a time
            # and makes a prediction based on that. Then selects the best
            # exogenous column at the end.
            y_train = data.iloc[:, [0, d_val]]
            print('\nAdditional Variable in VAR model = %s' % cols[d_val])
            info_criteria = pd.DataFrame(
                index=['AR{}'.format(i) for i in range(0, self.p_max+1)],
                columns=['MA{}'.format(i) for i in range(0, self.q_max+1)]
            )
            for p_val, q_val in itertools.product(range(0, self.p_max+1), range(0, self.q_max+1)):
                if p_val == 0 and q_val == 0:
                    info_criteria.loc['AR{}'.format(p_val), 'MA{}'.format(q_val)] = np.nan
                    print(' Iteration %d completed' % i)
                    i += 1
                else:
                    try:
                        model = VARMAX(y_train, order=(p_val, q_val), trend='c')
                        model = model.fit(max_iter=1000, disp=False)
                        info_criteria.loc['AR{}'.format(p_val), 'MA{}'.format(q_val)] = eval('model.' + self.scoring)
                        print(' Iteration %d completed' % i)
                        i += 1
                    except Exception:
                        i += 1
                        print(' Iteration %d completed' % i)
            info_criteria = info_criteria[info_criteria.columns].astype(float)
            interim_d = copy.deepcopy(d_val)
            interim_p, interim_q, interim_bic = find_lowest_pq(info_criteria)
            if self.verbose == 1:
                _, axis = plt.subplots(figsize=(20, 10))
                axis = sns.heatmap(
                    info_criteria,
                    mask=info_criteria.isnull(),
                    ax=axis,
                    annot=True,
                    fmt='.0f'
                )
                axis.set_title(self.scoring)
            results_dict[str(interim_p) + ' ' + str(interim_d) + ' ' + str(interim_q)] = interim_bic
        best_bic = min(results_dict.items(), key=operator.itemgetter(1))[1]
        best_pdq = min(results_dict.items(), key=operator.itemgetter(1))[0]
        self.best_p = int(best_pdq.split(' ')[0])
        self.best_d = int(best_pdq.split(' ')[1])
        self.best_q = int(best_pdq.split(' ')[2])
        
        print('Best variable selected for VAR: %s' % data.columns.tolist()[self.best_d])

    def refit(self, ts_df: pd.DataFrame) -> object:
        """
        Refits an already trained model using a new dataset
        Useful when fitting to the full data after testing with cross validation
        :param ts_df The time series data to be used for fitting the model
        :type ts_df pd.DataFrame
        :rtype object
        """
        bestmodel = self.get_best_model(ts_df)
        print('Refitting data with previously found best parameters')
        try:
            self.model = bestmodel.fit(disp=False)
            print('    Best %s metric = %0.1f' % (self.scoring, eval('self.model.' + self.scoring)))
        except Exception as exception:
            print(exception)

        return self


    def get_best_model(self, data: pd.DataFrame):
        """
        Returns the 'unfit' SARIMAX model with the given dataset and the
        selected best parameters. This can be used to fit or refit the model.
        """
        bestmodel = VARMAX(data, order=(self.best_p, self.best_q), trend='c')
        return bestmodel
