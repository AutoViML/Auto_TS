import math

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from pmdarima.arima.auto import auto_arima # type: ignore

from tscv import GapWalkForward # type: ignore

# # imported SARIMAX from statsmodels pkg
# from statsmodels.tsa.statespace.sarimax import SARIMAX  # type: ignore

from pmdarima.arima import auto_arima # type: ignore

from ..build_base import BuildBase
from .build_arima_base import BuildArimaBase

# helper functions
from ...utils import colorful, print_static_rmse, print_dynamic_rmse
from ...models.ar_based.param_finder import find_best_pdq_or_PDQ


class BuildAutoSarimax(BuildArimaBase):
    
    def find_best_parameters(self, data: pd.DataFrame):
        """
        Given a dataset, finds the best parameters using the settings in the class
        """

        if self.verbose >= 1:
            print(colorful.BOLD + '\n    Finding the best parameters using AutoArima:' + colorful.END)
        arima_model =  auto_arima(
            y = data[self.original_target_col],
            # exogenous=data[self.original_preds],  # TODO: Check if this should be included
            out_of_sample_size=0,  # use whole dataset to compute metrics
            information_criterion=self.scoring, # AIC
            scoring='mse', # only supports 'mse' or 'mae'

            # TODO: Check if we can go higher on max p and q (till seasnonality)
            start_p=0, d=None, start_q=0, max_p=self.p_max, max_d=self.d_max, max_q=self.q_max, # AR Parameters
            start_P=0, D=None, start_Q=0, max_P=self.p_max, max_D=self.d_max, max_Q=self.q_max, # Seasonal Parameters (1) 
            m=self.seasonal_period, seasonal=self.seasonality, # Seasonal Parameters (2) 
            stepwise = True, random_state=42, n_fits = 50, n_jobs=1,  # Hyperparameer Search
            error_action='warn', trace = True, supress_warnings=True 
        )

        self.best_p, self.best_d, self.best_q = arima_model.order  # example (0, 1, 1)
        self.best_P, self.best_D, self.best_Q, _ = arima_model.seasonal_order # example (2, 1, 1, 12)

        metric_value = math.nan

        if self.scoring.lower() == 'aic':
            metric_value = arima_model.aic()
        elif self.scoring.lower() == 'aicc':
            metric_value = arima_model.aicc()
        elif self.scoring.lower() == 'bic':
            metric_value = arima_model.bic()
        else:
            print("Error: Metric provided is not correct. Supports one of 'aic', 'aicc', or 'bic'")
        
        if self.verbose >= 1:
            print(
                '\nBest model is a Seasonal SARIMAX(%d,%d,%d)*(%d,%d,%d,%d), %s = %0.3f' % (
                self.best_p, self.best_d, self.best_q,
                self.best_P, self.best_D, self.best_Q,
                self.seasonal_period, self.scoring, metric_value)
            )
              
  