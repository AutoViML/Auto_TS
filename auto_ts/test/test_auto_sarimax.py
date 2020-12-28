"""
Unit Tests for BuildAutoSarimax

----------------------
Total Combinations: 8
----------------------
Seasonality: Seasonal, Non-Seasonal (2)
Univariate, Multivariate (2)
CV: Yes, No (2)
"""

import unittest
import numpy as np # type: ignore
import pandas as pd # type: ignore

from pandas.testing import assert_series_equal # type: ignore
from pandas.testing import assert_frame_equal # type: ignore
from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper  # type: ignore


class TestAutoSarimax(unittest.TestCase):

    def setUp(self):
        # Pre Release
        import sys
        import os
        sys.path.append(os.environ['DEV_AUTOTS'])
        import pandas as pd  # type: ignore

        datapath = 'example_datasets/'
        filename1 = 'Sales_and_Marketing.csv'
        dft = pd.read_csv(datapath+filename1,index_col=None)

        self.ts_column = 'Time Period'
        self.sep = ','
        self.target = 'Sales'
        self.preds = [x for x in list(dft) if x not in [self.ts_column, self.target]] # Exogenous variable names

        self.train_multivar = dft[:40]
        self.test_multivar = dft[40:]

        self.train_univar = dft[:40][[self.ts_column, self.target]]
        self.test_univar = dft[40:][[self.ts_column, self.target]]

        self.forecast_period = 8

        self.expected_pred_col_names = np.array(['mean', 'mean_se', 'mean_ci_lower', 'mean_ci_upper'])

        ########################
        #### Golden Results ####
        ########################

        # TODO: Add to each individual test
        ## For each of the 8 combinations, we need the following
        # Internal Validation results (for each fold)
        # Internal Validation RMSE (overall and for each fold)

        # External Test results (various combinations of prediction windows - same as forecast period OR not same)
        # External Test RMSE


    def test_seasonal_univar_noCV(self):
        """
        Test 1: Seasonal Univariate Without CV
        """
        pass

    def test_seasonal_univar_CV(self):
        """
        Test 2: Seasonal Univariate With CV
        """
        pass

    def test_seasonal_multivar_noCV(self):
        """
        Test 3: Seasonal Multivariate Without CV
        """
        pass

    def test_seasonal_multivar_CV(self):
        """
        Test 4: Seasonal Multivariate With CV
        """
        pass

    def test_nonseasonal_univar_noCV(self):
        """
        Test 5: Non Seasonal Univariate Without CV
        """
        pass

    def test_nonseasonal_univar_CV(self):
        """
        Test 6: Non Seasonal Univariate With CV
        """
        pass

    def test_nonseasonal_multivar_noCV(self):
        """
        Test 7: Non Seasonal Multivariate Without CV
        """
        pass

    def test_nonseasonal_multivar_CV(self):
        """
        Test 8: Non Seasonal Multivariate With CV
        """
        pass



