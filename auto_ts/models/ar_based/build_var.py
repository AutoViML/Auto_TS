"""Module to build a VAR model
"""

from typing import Optional
import warnings
import itertools
import operator
import copy

import numpy as np  # type: ignore
import pandas as pd # type: ignore
from pandas.core.generic import NDFrame # type:ignore

import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
sns.set(style="white", color_codes=True)

from statsmodels.tsa.statespace.varmax import VARMAX # type: ignore

# helper functions
from ...utils import print_dynamic_rmse
from ...models.ar_based.param_finder import find_lowest_pq


class BuildVAR():
    """Class to build a VAR model
    """
    def __init__(self, criteria, forecast_period=2, p_max=3, q_max=3, verbose=0):
        """
        Automatically build a VAR Model
        """
        self.criteria = criteria
        self.forecast_period = forecast_period
        self.p_max = p_max
        self.q_max = q_max
        self.verbose = verbose
        self.model = None
        self.best_p = None
        self.best_d = None
        self.best_q = None

    def fit(self, ts_df):
        """
        This builds a VAR model given a multivariate time series data frame with time as the Index.
        Note that the input "y_train" can be a data frame with one column or multiple cols or a
        multivariate array. However, the first column must be the target variable. The others are added.
        You must include only Time Series data in it. DO NOT include "Non-Stationary" or "Trendy" data.
        Make sure your Time Series is "Stationary" before you send it in!! If not, this will give spurious
        results. Since it automatically builds a VAR model, you need to give it a Criteria to optimize on.
        You can give it any of the following metrics as criteria: AIC, BIC, Deviance, Log-likelihood.
        You can give the highest order values for p and q. Default is set to 3 for both.
        """

        ts_df = ts_df[:]

        self.find_best_parameters(data = ts_df)

        # Cross Validate (original)
        ts_train = ts_df[:-self.forecast_period]
        ts_test = ts_df[-self.forecast_period:]

        if self.verbose == 1:
            print(
                'Data Set split into train %s and test %s for Cross Validation Purposes'
                % (ts_train.shape, ts_test.shape)
            )

        y_train = ts_train.iloc[:, [0, self.best_d]]
        self.model = VARMAX(y_train, order=(self.best_p, self.best_q), trend='c')
        self.model = self.model.fit(disp=False)
        if self.verbose == 1:
            self.model.plot_diagnostics(figsize=(16, 12))
            axis = self.model.impulse_responses(12, orthogonalized=True).plot(figsize=(12, 4))
            axis.set(xlabel='Time Steps', title='Impulse Response Functions')

        res_df = self.predict(simple=False)

        rmse, norm_rmse = print_dynamic_rmse(ts_test.iloc[:, 0], res_df['mean'].values, ts_train.iloc[:, 0])
        return self.model, res_df, rmse, norm_rmse

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

        # Extract the dynamic predicted and true values of our time series
        if forecast_period is None:
            # use the forecast period used during training
            forecast_period = self.forecast_period

        # y_forecasted = self.model.forecast(forecast_period)

        res = self.model.get_forecast(forecast_period)
        res_frame = res.summary_frame()

        if simple:
            res_frame = res_frame['mean']
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
                        info_criteria.loc['AR{}'.format(p_val), 'MA{}'.format(q_val)] = eval('model.' + self.criteria)
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
                axis.set_title(self.criteria)
            results_dict[str(interim_p) + ' ' + str(interim_d) + ' ' + str(interim_q)] = interim_bic
        best_bic = min(results_dict.items(), key=operator.itemgetter(1))[1]
        best_pdq = min(results_dict.items(), key=operator.itemgetter(1))[0]
        self.best_p = int(best_pdq.split(' ')[0])
        self.best_d = int(best_pdq.split(' ')[1])
        self.best_q = int(best_pdq.split(' ')[2])
        print('Best variable selected for VAR: %s' % data.columns.tolist()[self.best_d])
