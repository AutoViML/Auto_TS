"""
Unit Tests for VAR Models

----------------------
Total Combinations: 4
----------------------
Seasonality: NA
Univariate, Multivariate: Simple Independent Test for Univariate (1)
CV: Yes, No (2)
"""

import sys
import os
import unittest
import math
import numpy as np # type: ignore
import pandas as pd # type: ignore

from pandas.testing import assert_series_equal # type: ignore
from pandas.testing import assert_frame_equal # type: ignore
from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper  # type: ignore

sys.path.append(os.environ['DEV_AUTOTS'])
from auto_ts import auto_timeseries as ATS

class TestVAR(unittest.TestCase):

    def setUp(self):
        # Pre Release


        datapath = 'example_datasets/'
        filename1 = 'Sales_and_Marketing.csv'
        dft = pd.read_csv(datapath + filename1, index_col = None)

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


        ############################
        #### VAR Golden Results ####
        ############################

        #### UNIVARIATE ####
        self.forecast_gold_var_univar = None
        self.rmse_gold_var_univar = math.inf
        self.forecast_gold_var_univar_series = None
        self.forecast_gold_var_univar_series_10 = None

        #### MULTIVARIATE ####

        # Internal (to AutoML) validation set results
        self.forecast_gold_var_multivar_internal_val_cv_fold1 = np.array([
            510.302336, 531.109224, 536.878513, 534.311164,
            529.305887, 525.199071, 523.015255, 522.445215
        ])

        self.forecast_gold_var_multivar_internal_val_cv_fold2 = np.array([
            741.377909, 676.233419, 615.538721, 571.797729,
            546.952783, 537.342231, 537.474487, 542.307393
        ])

        self.rmse_gold_var_multivar_cv_fold1 = 155.21757611
        self.rmse_gold_var_multivar_cv_fold2 = 112.4770318 # Without CV gets this result

        ## External Test Set results
        results = [
            675.899931, 622.204059, 578.38291, 553.067517,
            543.612945, 543.696406, 547.604403, 551.762352
            ]
        index = pd.to_datetime([
            '2014-05-01', '2014-06-01', '2014-07-01', '2014-08-01',
            '2014-09-01', '2014-10-01', '2014-11-01', '2014-12-01'
            ])
        self.forecast_gold_var_multivar = np.array(results)

        self.forecast_gold_var_multivar_series = pd.Series(data=results, index=index)
        self.forecast_gold_var_multivar_series.name = 'mean'

        results = results + [554.643756, 556.055009]
        index = pd.to_datetime([
            '2014-05-01', '2014-06-01', '2014-07-01', '2014-08-01',
            '2014-09-01', '2014-10-01', '2014-11-01', '2014-12-01',
            '2015-01-01', '2015-02-01'
            ])
        self.forecast_gold_var_multivar_series_10 = pd.Series(data=results, index=index)
        self.forecast_gold_var_multivar_series_10.name = 'mean'


    def test_noCV(self):
        """
        Test 1: VAR without CV
        """
        print("\n\n" + "*"*50)
        print("Performing Unit Test: 'test_noCV'")
        print("*"*50 + "\n\n")

        automl_model = ATS(
            score_type='rmse', forecast_period=self.forecast_period, time_interval='Month',
            non_seasonal_pdq=None, seasonality=False, seasonal_period=12,
            model_type='VAR',
            verbose=0)
        automl_model.fit(
            traindata=self.train_multivar,
            ts_column=self.ts_column,
            target=self.target,
            cv=None,
            sep=self.sep)

        ml_dict = automl_model.get_ml_dict()

        ######################
        ## External Results ##
        ######################

        # Simple forecast with forecast window = the one used in training
        # Using named model
        test_predictions = automl_model.predict(
            testdata=self.test_multivar[[self.ts_column] + self.preds], # Not needed for VAR
            forecast_period=self.forecast_period,
            model="VAR"
        )
        assert_series_equal(
            test_predictions['mean'].round(6),
            self.forecast_gold_var_multivar_series
        )

        # Simple forecast with forecast window != the one used in training
        # Using named model
        test_predictions = automl_model.predict(
            testdata=self.test_multivar[[self.ts_column] + self.preds], # Not needed for VAR
            forecast_period=10,
            model="VAR"
        )
        assert_series_equal(test_predictions['mean'].round(6), self.forecast_gold_var_multivar_series_10)

        # Complex forecasts (returns confidence intervals, etc.)
        test_predictions = automl_model.predict(
            testdata=self.test_multivar[[self.ts_column] + self.preds], # Not needed for VAR
            forecast_period=self.forecast_period,
            model="VAR",
            simple=False
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                test_predictions.columns.values, self.expected_pred_col_names
            )
        )

        ###################
        ## ML Dictionary ##
        ###################
        self.assertIsNone(
            np.testing.assert_array_equal(
                np.round(ml_dict.get('VAR').get('forecast')[0]['mean'].values.astype(np.double), 6),
                self.forecast_gold_var_multivar_internal_val_cv_fold2
            ),
            "(Multivar Test) VAR Forecast does not match up with expected values."
        )

        self.assertEqual(
            round(ml_dict.get('VAR').get('rmse')[0], 8), self.rmse_gold_var_multivar_cv_fold2,
            "(Multivar Test) VAR RMSE does not match up with expected values.")

    def test_CV(self):
        """
        Test 2: VAR with CV
        """
        print("\n\n" + "*"*50)
        print("Performing Unit Test: 'test_CV'")
        print("*"*50 + "\n\n")

        automl_model = ATS(
            score_type='rmse', forecast_period=self.forecast_period, time_interval='Month',
            non_seasonal_pdq=None, seasonality=False, seasonal_period=12,
            model_type='VAR',
            verbose=0)
        automl_model.fit(
            traindata=self.train_multivar,
            ts_column=self.ts_column,
            target=self.target,
            cv=2,
            sep=self.sep)

        ml_dict = automl_model.get_ml_dict()

        ######################
        ## External Results ##
        ######################

        # Simple forecast with forecast window = the one used in training
        # Using named model
        test_predictions = automl_model.predict(
            testdata=self.test_multivar[[self.ts_column] + self.preds], # Not needed for VAR
            forecast_period=self.forecast_period,
            model="VAR"
        )
        assert_series_equal(
            test_predictions['mean'].round(6),
            self.forecast_gold_var_multivar_series
        )

        # Simple forecast with forecast window != the one used in training
        # Using named model
        test_predictions = automl_model.predict(
            testdata=self.test_multivar[[self.ts_column] + self.preds], # Not needed for VAR
            forecast_period=10,
            model="VAR"
        )
        assert_series_equal(test_predictions['mean'].round(6), self.forecast_gold_var_multivar_series_10)

        # Complex forecasts (returns confidence intervals, etc.)
        test_predictions = automl_model.predict(
            testdata=self.test_multivar[[self.ts_column] + self.preds], # Not needed for VAR
            forecast_period=self.forecast_period,
            model="VAR",
            simple=False
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                test_predictions.columns.values, self.expected_pred_col_names
            )
        )

        ###################
        ## ML Dictionary ##
        ###################
        self.assertIsNone(
            np.testing.assert_array_equal(
                np.round(ml_dict.get('VAR').get('forecast')[0]['mean'].values.astype(np.double), 6),
                self.forecast_gold_var_multivar_internal_val_cv_fold1,

            ),
            "(Multivar Test) VAR Forecast does not match up with expected values."
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                np.round(ml_dict.get('VAR').get('forecast')[1]['mean'].values.astype(np.double), 6),
                self.forecast_gold_var_multivar_internal_val_cv_fold2
            ),
            "(Multivar Test) VAR Forecast does not match up with expected values."
        )

        self.assertEqual(
            round(ml_dict.get('VAR').get('rmse')[0], 8), self.rmse_gold_var_multivar_cv_fold1,
            "(Multivar Test) VAR RMSE does not match up with expected values.")
        self.assertEqual(
            round(ml_dict.get('VAR').get('rmse')[1], 8), self.rmse_gold_var_multivar_cv_fold2,
            "(Multivar Test) VAR RMSE does not match up with expected values.")


    def test_univar(self):
        """
        Test 3: Univariate VAR
        """
        print("\n\n" + "*"*50)
        print("Performing Unit Test: 'test_univar'")
        print("*"*50 + "\n\n")

        automl_model = ATS(
            score_type='rmse', forecast_period=self.forecast_period, time_interval='Month',
            non_seasonal_pdq=None, seasonality=False, seasonal_period=12,
            model_type='VAR',
            verbose=0)
        automl_model.fit(
            traindata=self.train_univar,
            ts_column=self.ts_column,
            target=self.target,
            cv=None
            )
        ml_dict = automl_model.get_ml_dict()

        self.assertIsNone(automl_model.get_model_build('VAR'), "Expected Univar VAR model to be None but did not get None.")

        # Simple forecast with forecast window = one used in training
        # Using named model
        test_predictions = automl_model.predict(
            forecast_period=self.forecast_period,
            model="VAR"
        )
        self.assertIsNone(test_predictions)

        # Simple forecast with forecast window != one used in training
        # Using named model
        test_predictions = automl_model.predict(
            forecast_period=10,
            model="VAR"
        )
        self.assertIsNone(test_predictions)

        # Complex forecasts (returns confidence intervals, etc.)
        test_predictions = automl_model.predict(
            forecast_period=self.forecast_period,
            model="VAR",
            simple=False
        )
        self.assertIsNone(test_predictions)

        ###################
        ## ML Dictionary ##
        ###################
        self.assertEqual(
            ml_dict.get('VAR').get('forecast'), self.forecast_gold_var_univar,
            "(Univar Test) VAR Forecast does not match up with expected values."
        )

        self.assertEqual(
            round(ml_dict.get('VAR').get('rmse'), 8), self.rmse_gold_var_univar,
            "(Univar Test) VAR RMSE does not match up with expected values.")