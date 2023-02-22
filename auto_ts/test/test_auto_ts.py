"""Module to test the AutoML flow
TODO: Make this code more modular and like unit tests
Currently, it runs like regression tests
Separate out the testing of individual models into their own modules
like what has been done for auto_SARIMAX and change this module to
focus purely on testing the AutoML flow
"""

import sys
import os
import unittest
import math
import numpy as np # type: ignore
import pandas as pd # type: ignore

from pandas.testing import assert_series_equal # type: ignore
from pandas.testing import assert_frame_equal # type: ignore
from prophet.forecaster import Prophet # type: ignore
from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper  # type: ignore
from sklearn.base import BaseEstimator

sys.path.append(os.environ['DEV_AUTOTS'])
from auto_ts import auto_timeseries as ATS

class TestAutoTS(unittest.TestCase):
    """Class to test the AutoML flow
    """
    def setUp(self):
        # Pre Release

        datapath = 'example_datasets/'
        filename1 = 'Sales_and_Marketing.csv'
        dft = pd.read_csv(datapath+filename1, index_col=None)

        self.ts_column = 'Time Period'
        self.sep = ','
        self.target = 'Sales'
        self.preds = [x for x in list(dft) if x not in [self.ts_column, self.target]]  # Exogenous variable names

        self.train_multivar = dft[:40]
        self.test_multivar = dft[40:]

        self.train_univar = dft[:40][[self.ts_column, self.target]]
        self.test_univar = dft[40:][[self.ts_column, self.target]]

        self.forecast_period = 8

        # TODO: Update Prophet column names when you make it consistent
        # self.expected_pred_col_names_prophet = np.array(['mean', 'mean_se', 'mean_ci_lower', 'mean_ci_upper'])
        self.expected_pred_col_names = np.array(['mean', 'mean_se', 'mean_ci_lower', 'mean_ci_upper'])

        ################################
        #### Prophet Golden Results ####
        ################################


        #########################################################
        #### UNIVARIATE [Both No CV (uses fold2) and CV = 2] ####
        #########################################################

        ## Internal (to AutoML) validation set results
        self.forecast_gold_prophet_univar_internal_val_cv_fold1 = np.array([
            # Previous Unbiased Estimate which was giving issues when tested
            # against some time series
            # 447.733878, 510.152411, 536.119254, 599.663702,
            # 592.020981, 657.844486, 667.245238, 485.989273

            # Biased estimate since it does not match other models in terms of
            # validation period or count of observations in each period
            380.824105, 513.599456, 509.112615, 534.484328, 565.162594,
            633.571696, 628.649043, 700.044705, 713.049146, 531.755138,
            468.986024, 431.764757, 432.869342
        ])

        self.forecast_gold_prophet_univar_internal_val_cv_fold2 = np.array([
            614.482071, 562.050462, 534.810663, 605.566298,
            580.899233, 585.676464, 686.480721, 732.167184
        ])

        # Previous Unbiased Estimate which was giving issues when tested
        # against some time series
        # self.rmse_gold_prophet_univar_cv_fold1 = 86.34827037

        # Biased estimate since it does not match other models in terms of
        # validation period or count of observations in each period
        self.rmse_gold_prophet_univar_cv_fold1 = 67.12444991

        # Previous Unbiased Estimate which was giving issues when tested
        # against some time series
        # self.rmse_gold_prophet_univar_cv_fold2 = 56.5751 # Without CV gets this result

        # Biased estimate since it does not match other models in terms of
        # validation period or count of observations in each period
        self.rmse_gold_prophet_univar_cv_fold2 = 55.594384 # Without CV gets this result


        ## External Test Set results
        results = [
            749.061242, 751.077262, 796.892366, 783.206733,
            689.698130, 595.713426, 569.486600, 635.884371
            ]
        index = np.arange(40, 48)

        self.forecast_gold_prophet_univar_external_test_cv = pd.Series(
            data=results,
            index=index
        )
        self.forecast_gold_prophet_univar_external_test_cv.name = 'yhat'

        results = results + [576.473786, 581.275889]
        index = np.arange(40, 50)

        self.forecast_gold_prophet_univar_external_test_10_cv = pd.Series(
            data=results,
            index=index
        )
        self.forecast_gold_prophet_univar_external_test_10_cv.name = 'yhat'


        ###########################################################
        #### MULTIVARIATE [Both No CV (uses fold2) and CV = 2] ####
        ###########################################################

        # Internal (to AutoML) validation set results
        self.forecast_gold_prophet_multivar_internal_val_cv_fold1 = np.array([
            # Previous Unbiased Estimate which was giving issues when tested
            # against some time series
            # 502.111972, 569.181958, 578.128706, 576.069791,
            # 663.258686, 677.851419, 750.972617, 781.269791

            # Biased estimate since it does not match other models in terms of
            # validation period or count of observations in each period
            263.278655, 515.858752, 596.916275, 528.986198, 500.662700,
            598.963783, 618.888170, 689.703134, 729.021092, 512.653123,
            482.081110, 455.731872, 309.154906
        ])

        self.forecast_gold_prophet_multivar_internal_val_cv_fold2 = np.array([
            # Previous Unbiased Estimate which was giving issues when tested
            # against some time series
            # 618.244315, 555.784628, 524.396122, 611.513751,
            # 584.936717, 605.940656, 702.652641, 736.639273

            # Biased estimate since it does not match other models in terms of
            # validation period or count of observations in each period
            566.442657, 637.740324, 636.940975, 708.781598, 725.327137,
            542.517575, 484.142209, 449.074053, 505.764104, 479.320556,
            475.915244, 570.814914, 600.585457
        ])

        # Previous Unbiased Estimate which was giving issues when tested
        # against some time series
        # self.rmse_gold_prophet_multivar_cv_fold1 = 48.70419901

        # Biased estimate since it does not match other models in terms of
        # validation period or count of observations in each period
        self.rmse_gold_prophet_multivar_cv_fold1 = 116.40578011

        # Previous Unbiased Estimate which was giving issues when tested
        # against some time series
        # self.rmse_gold_prophet_multivar_cv_fold2 = 63.24631835 # Without CV gets this result

        # Biased estimate since it does not match other models in terms of
        # validation period or count of observations in each period
        self.rmse_gold_prophet_multivar_cv_fold2 = 53.60487935 # Without CV gets this result (for unbiased case, but not for biased case)


        ## External Test Set results
        results = [
            747.964093, 736.512241, 814.840792, 825.152970,
            657.743450, 588.985816, 556.814528, 627.768202
            ]
        index = np.arange(0, 8)

        self.forecast_gold_prophet_multivar_external_test_cv = pd.Series(
            data=results,
            index=index
        )
        self.forecast_gold_prophet_multivar_external_test_cv.name = 'yhat'

        # Same as regular since we only have 8 exogenous observations
        self.forecast_gold_prophet_multivar_external_test_10_cv = pd.Series(
            data=results,
            index=index
        )
        self.forecast_gold_prophet_multivar_external_test_10_cv.name = 'yhat'


        # ##############################
        # #### ARIMA Golden Results ####
        # ##############################

        # #### UNIVARIATE and MULTIVARIATE ####

        # results = [
        #     801.78660584, 743.16044526, 694.38764549, 684.72931967,
        #     686.70229610, 692.13402266, 698.59426282, 705.36034762
        #     ]
        # index = [
        #     'Forecast_1', 'Forecast_2', 'Forecast_3', 'Forecast_4',
        #     'Forecast_5', 'Forecast_6', 'Forecast_7', 'Forecast_8'
        #     ]

        # self.forecast_gold_arima_uni_multivar = np.array(results)

        # self.forecast_gold_arima_uni_multivar_series = pd.Series(
        #     data=results,
        #     index=index
        # )
        # self.forecast_gold_arima_uni_multivar_series.name = 'mean'

        # results = results + [712.217380, 719.101457]
        # index = index + ['Forecast_9', 'Forecast_10']

        # self.forecast_gold_arima_uni_multivar_series_10 = pd.Series(
        #     data=results,
        #     index=index
        # )
        # self.forecast_gold_arima_uni_multivar_series_10.name = 'mean'

        # self.rmse_gold_arima_uni_multivar = 169.00016628


        # #######################################################################################################

        # ################################
        # #### SARIMAX Golden Results ####
        # ################################

        # #### UNIVARIATE ####

        # ## Internal (to AutoML) validation set results
        # results = [
        #     803.31673726, 762.46093997, 718.35819310, 711.42130506,
        #     719.36254603, 732.70981867, 747.57645435, 762.47349398
        #     ]
        # self.forecast_gold_sarimax_univar_internal_val = np.array(results)
        # self.rmse_gold_sarimax_univar = 193.49650578

        # ## External Test Set results
        # results = [
        #     737.281499, 718.144765, 672.007487, 618.321458,
        #     578.990868, 567.799468, 586.467414, 625.619993
        # ]
        # index = pd.to_datetime([
        #     '2014-05-01', '2014-06-01', '2014-07-01', '2014-08-01',
        #     '2014-09-01', '2014-10-01', '2014-11-01', '2014-12-01'
        #     ])

        # self.forecast_gold_sarimax_univar_external_test = pd.Series(
        #     data=results,
        #     index=index
        # )
        # self.forecast_gold_sarimax_univar_external_test.name = 'mean'

        # results = results + [669.666326, 703.29552]
        # index = pd.to_datetime([
        #     '2014-05-01', '2014-06-01', '2014-07-01', '2014-08-01',
        #     '2014-09-01', '2014-10-01', '2014-11-01', '2014-12-01',
        #     '2015-01-01', '2015-02-01'
        #     ])

        # self.forecast_gold_sarimax_univar_external_test_10 = pd.Series(
        #     data=results,
        #     index=index
        # )
        # self.forecast_gold_sarimax_univar_external_test_10.name = 'mean'


        # #######################################
        # #### MULTIVARIATE (no seasonality) ####
        # #######################################

        # ## Internal (to AutoML) validation set results
        # results = [
        #     772.268886, 716.337431, 686.167231, 739.269047,
        #     704.280567, 757.450733, 767.711055, 785.960125
        # ]
        # self.forecast_gold_sarimax_multivar_internal_val = np.array(results)
        # self.rmse_gold_sarimax_multivar = 185.704684

        # ## External Test Set results (With Multivariate columns accepted)
        # results = [
        #     750.135204, 806.821297, 780.232195, 743.309074,
        #     724.400616, 683.117893, 673.696113, 686.807075
        # ]
        # index = pd.to_datetime([
        #     '2014-05-01', '2014-06-01', '2014-07-01', '2014-08-01',
        #     '2014-09-01', '2014-10-01', '2014-11-01', '2014-12-01'
        #     ])

        # self.forecast_gold_sarimax_multivar_external_test = pd.Series(
        #     data=results,
        #     index=index
        # )
        # self.forecast_gold_sarimax_multivar_external_test.name = 'mean'

        # results = results[0:6]
        # index = index[0:6]
        # self.forecast_gold_sarimax_multivar_external_test_10 = pd.Series(
        #     data=results,
        #     index=index
        # )
        # self.forecast_gold_sarimax_multivar_external_test_10.name = 'mean'

        # ################################################################################
        # #### MULTIVARIATE (with seasonality = True, Seasonal Period = 12 CV = None) ####
        # ################################################################################

        # ## Internal (to AutoML) validation set results (with seasonality = True, Seasonal Period = 12)
        # # Without CV
        # results = [
        #     726.115602, 646.028979, 657.249936, 746.752393,
        #     732.813245, 749.435178, 863.356789, 903.168728
        # ]
        # self.forecast_gold_sarimax_multivar_internal_val_s12 = np.array(results)
        # self.rmse_gold_sarimax_multivar_s12 = 197.18894

        # ## External Test Set results
        # # (With Multivariate columns accepted)
        # # (with seasonality = True, Seasonal Period = 12)

        # results = [
        #     1006.134134, 779.874076, 420.461804, 724.042104,
        #     1827.304601, 1204.070838, -2216.439611, -1278.974132
        # ]

        # index = pd.to_datetime([
        #     '2014-05-01', '2014-06-01', '2014-07-01', '2014-08-01',
        #     '2014-09-01', '2014-10-01', '2014-11-01', '2014-12-01'
        #     ])

        # self.forecast_gold_sarimax_multivar_external_test_s12 = pd.Series(
        #     data=results,
        #     index=index
        # )
        # self.forecast_gold_sarimax_multivar_external_test_s12.name = 'mean'

        # results = results[0:6]
        # index = index[0:6]
        # self.forecast_gold_sarimax_multivar_external_test_10_s12 = pd.Series(
        #     data=results,
        #     index=index
        # )
        # self.forecast_gold_sarimax_multivar_external_test_10_s12.name = 'mean'

        # #############################################################################
        # #### MULTIVARIATE (with seasonality = True, Seasonal Period = 3, CV = 2) ####
        # #############################################################################

        # ## Internal (to AutoML) validation set results

        # results = [
        #     119.260686, 540.623654, 230.040446, 364.088969,
        #     470.581971, 105.559723, 84.335069, 110.757574
        # ]
        # self.forecast_gold_sarimax_multivar_internal_val_s3_cv_fold1 = np.array(results)

        # results = [
        #     551.736392, 502.232401, 440.047123, 521.382176,
        #     496.012325, 501.563083, 634.825011, 674.975611
        # ]
        # self.forecast_gold_sarimax_multivar_internal_val_s3_cv_fold2 = np.array(results)

        # self.rmse_gold_sarimax_multivar_s12_cv = 239.191102
        # self.rmse_gold_sarimax_multivar_s3_cv_fold1 = 443.839435
        # self.rmse_gold_sarimax_multivar_s3_cv_fold2 = 34.542769


        # ## External Test Set results
        # results = [
        #     770.447134, 784.881945, 857.496478, 918.626627,
        #     689.107408, 599.827292, 608.747367, 634.957579
        # ]

        # index = pd.to_datetime([
        #     '2014-05-01', '2014-06-01', '2014-07-01', '2014-08-01',
        #     '2014-09-01', '2014-10-01', '2014-11-01', '2014-12-01'
        #     ])

        # self.forecast_gold_sarimax_multivar_external_test_s3_cv = pd.Series(
        #     data=results,
        #     index=index
        # )
        # self.forecast_gold_sarimax_multivar_external_test_s3_cv.name = 'mean'

        # results = results[0:6]
        # index = index[0:6]
        # self.forecast_gold_sarimax_multivar_external_test_10_s3_cv = pd.Series(
        #     data=results,
        #     index=index
        # )
        # self.forecast_gold_sarimax_multivar_external_test_10_s3_cv.name = 'mean'

        #######################################################################################################

        #####################################
        #### Auto SARIMAX Golden Results ####
        #####################################

        #### UNIVARIATE ####

        ## Internal (to AutoML) validation set results
        # results = [
        #     803.31673726, 762.46093997, 718.3581931,  711.42130506,
        #     719.36254603, 732.70981867, 747.57645435, 762.47349398
        #     ]
        # self.forecast_gold_sarimax_univar_internal_val = np.array(results)
        self.rmse_gold_auto_sarimax_univar = 128.034697

        ## External Test Set results
        # results=[
        #     737.281499, 718.144765, 672.007487, 618.321458,
        #     578.990868, 567.799468, 586.467414, 625.619993
        # ]
        # index = pd.to_datetime([
        #     '2014-05-01', '2014-06-01', '2014-07-01', '2014-08-01',
        #     '2014-09-01', '2014-10-01', '2014-11-01', '2014-12-01'
        #     ])

        # self.forecast_gold_sarimax_univar_external_test = pd.Series(
        #         data = results,
        #         index = index
        #     )
        # self.forecast_gold_sarimax_univar_external_test.name = 'mean'

        # results = results + [669.666326, 703.29552]
        # index = pd.to_datetime([
        #     '2014-05-01', '2014-06-01', '2014-07-01', '2014-08-01',
        #     '2014-09-01', '2014-10-01', '2014-11-01', '2014-12-01',
        #     '2015-01-01', '2015-02-01'
        #     ])

        # self.forecast_gold_sarimax_univar_external_test_10 = pd.Series(
        #         data = results,
        #         index = index
        #     )
        # self.forecast_gold_sarimax_univar_external_test_10.name = 'mean'


        #######################################
        #### MULTIVARIATE (no seasonality) ####
        #######################################

        # ## Internal (to AutoML) validation set results
        # results = [
        #     772.268886, 716.337431, 686.167231, 739.269047,
        #     704.280567, 757.450733, 767.711055, 785.960125
        # ]
        # self.forecast_gold_sarimax_multivar_internal_val = np.array(results)
        self.rmse_gold_auto_sarimax_multivar = 147.703077

        ## External Test Set results (With Multivariate columns accepted)
        # results = [
        #     750.135204, 806.821297, 780.232195, 743.309074,
        #     724.400616, 683.117893, 673.696113, 686.807075
        # ]
        # index = pd.to_datetime([
        #     '2014-05-01', '2014-06-01', '2014-07-01', '2014-08-01',
        #     '2014-09-01', '2014-10-01', '2014-11-01', '2014-12-01'
        #     ])

        # self.forecast_gold_sarimax_multivar_external_test = pd.Series(
        #         data = results,
        #         index = index
        #     )
        # self.forecast_gold_sarimax_multivar_external_test.name = 'mean'

        # results = results[0:6]
        # index = index[0:6]
        # self.forecast_gold_sarimax_multivar_external_test_10 = pd.Series(
        #         data = results,
        #         index = index
        #     )
        # self.forecast_gold_sarimax_multivar_external_test_10.name = 'mean'

        ################################################################################
        #### MULTIVARIATE (with seasonality = True, Seasonal Period = 12 CV = None) ####
        ################################################################################

        ## Internal (to AutoML) validation set results (with seasonality = True, Seasonal Period = 12)
        # # Without CV
        # results = [
        #     726.115602, 646.028979, 657.249936, 746.752393,
        #     732.813245, 749.435178, 863.356789, 903.168728
        # ]
        # self.forecast_gold_sarimax_multivar_internal_val_s12 = np.array(results)
        self.rmse_gold_auto_sarimax_multivar_s12 = 90.083682

        ## External Test Set results (With Multivariate columns accepted)
        # (with seasonality = True, Seasonal Period = 12)

        # results = [
        #     1006.134134, 779.874076, 420.461804, 724.042104,
        #     1827.304601, 1204.070838, -2216.439611, -1278.974132
        # ]

        # index = pd.to_datetime([
        #     '2014-05-01', '2014-06-01', '2014-07-01', '2014-08-01',
        #     '2014-09-01', '2014-10-01', '2014-11-01', '2014-12-01'
        #     ])

        # self.forecast_gold_sarimax_multivar_external_test_s12 = pd.Series(
        #         data = results,
        #         index = index
        #     )
        # self.forecast_gold_sarimax_multivar_external_test_s12.name = 'mean'

        # results = results[0:6]
        # index = index[0:6]
        # self.forecast_gold_sarimax_multivar_external_test_10_s12 = pd.Series(
        #         data = results,
        #         index = index
        #     )
        # self.forecast_gold_sarimax_multivar_external_test_10_s12.name = 'mean'

        #############################################################################
        #### MULTIVARIATE (with seasonality = True, Seasonal Period = 3, CV = 2) ####
        #############################################################################

        ## Internal (to AutoML) validation set results

        # results = [
        #     119.260686, 540.623654, 230.040446, 364.088969,
        #     470.581971, 105.559723, 84.335069, 110.757574
        # ]
        # self.forecast_gold_sarimax_multivar_internal_val_s3_cv_fold1 = np.array(results)

        # results = [
        #     551.736392, 502.232401, 440.047123, 521.382176,
        #     496.012325, 501.563083, 634.825011, 674.975611
        # ]
        # self.forecast_gold_sarimax_multivar_internal_val_s3_cv_fold2 = np.array(results)

        self.rmse_gold_auto_sarimax_multivar_s12_cv = 268.191766
        # self.rmse_gold_auto_sarimax_multivar_s3_cv_fold1 = 443.839435
        # self.rmse_gold_auto_sarimax_multivar_s3_cv_fold2 = 34.542769


        # ## External Test Set results
        # results = [
        #     770.447134, 784.881945, 857.496478, 918.626627,
        #     689.107408, 599.827292, 608.747367, 634.957579
        # ]

        # index = pd.to_datetime([
        #     '2014-05-01', '2014-06-01', '2014-07-01', '2014-08-01',
        #     '2014-09-01', '2014-10-01', '2014-11-01', '2014-12-01'
        #     ])

        # self.forecast_gold_sarimax_multivar_external_test_s3_cv = pd.Series(
        #         data = results,
        #         index = index
        #     )
        # self.forecast_gold_sarimax_multivar_external_test_s3_cv.name = 'mean'

        # results = results[0:6]
        # index = index[0:6]
        # self.forecast_gold_sarimax_multivar_external_test_10_s3_cv = pd.Series(
        #         data = results,
        #         index = index
        #     )
        # self.forecast_gold_sarimax_multivar_external_test_10_s3_cv.name = 'mean'

        #######################################################################################################

        ############################
        #### VAR Golden Results ####
        ############################
        self.rmse_gold_var_multivar = 112.4770318


        ###########################
        #### ML Golden Results ####
        ###########################

        #### UNIVARIATE ####
        self.forecast_gold_ml_univar = None
        self.rmse_gold_ml_univar = math.inf
        self.forecast_gold_ml_univar_series = None
        self.forecast_gold_ml_univar_series_10 = None

        ##################################
        #### MULTIVARIATE (CV = None) ####
        ##################################

        ## Internal (to AutoML) validation set results
        # Initially only this was using rolling window
        # results = [
        #     509.64, 447.34, 438.2 , 456.98,
        #     453.04, 449.36, 530.02, 626.8
        # ]
        # Converted to using same as other model (no rolling window), but unable to return since
        # no internal test set (only validation sets in CV)
        results = []
        self.forecast_gold_ml_multivar_internal_val = results # np.array(results)
        # self.rmse_gold_ml_multivar = 74.133644 # Initially only this was using rolling window
        # self.rmse_gold_ml_multivar = 67.304009 # Converted to using same as other model (no rolling window)
        self.rmse_gold_ml_multivar = 71.824585 # With more engineered features (AutoViML)

        ## External Test Set results (With Multivariate columns accepted)
        # # Initially only this was using rolling window
        # results = [
        #     509.64, 485.24, 479.72, 483.98,
        #     482.78, 455.04, 518.62, 524.08
        # ]

        # Converted to using same as other model (no rolling window)
        results = [
            # 677.0, 652.85, 652.85, 652.85,
            # 640.458333, 505.043478, 494.571429, 494.571429

            # With more engineered features (AutoViML)
            # 733.293931, 627.633457, 621.182141, 614.128809,
            # 600.902623, 451.565462, 330.694427, 348.744604
            715.997991, 592.578743, 569.592338, 547.625097,
            513.972428, 334.603163, 188.705675, 180.323713
        ]
        # index = pd.RangeIndex(start=40, stop=48, step=1)
        # index = pd.to_datetime([
        index = [
            '2014-05-01', '2014-06-01', '2014-07-01', '2014-08-01',
            '2014-09-01', '2014-10-01', '2014-11-01', '2014-12-01'
            ]

        self.forecast_gold_ml_multivar_external_test = pd.Series(
            data=results,
            index=index
        )
        self.forecast_gold_ml_multivar_external_test.name = 'mean'
        self.forecast_gold_ml_multivar_external_test.index.name = self.ts_column

        results = results[0:6]
        index = index[0:6]
        #self.forecast_gold_ml_multivar_external_test_10_cv = pd.Series(
        self.forecast_gold_ml_multivar_external_test_10 = pd.Series(
            data=results,
            index=index
        )
        # self.forecast_gold_ml_multivar_external_test_10_cv.name = 'mean'
        self.forecast_gold_ml_multivar_external_test_10.name = 'mean'
        self.forecast_gold_ml_multivar_external_test_10.index.name = self.ts_column


        ###############################
        #### MULTIVARIATE (CV = 2) ####
        ###############################

        # self.rmse_gold_ml_multivar_cv = 80.167235
        # self.rmse_gold_ml_multivar_cv_fold1 = 93.03046227
        # self.rmse_gold_ml_multivar_cv_fold2 = 67.30400862

        # With more engineered features (AutoViML)
        self.rmse_gold_ml_multivar_cv = 94.472165
        self.rmse_gold_ml_multivar_cv_fold1 = 107.20370095
        self.rmse_gold_ml_multivar_cv_fold2 = 81.740629

        results = [
            # 677.000000, 652.850000, 652.850000, 652.850000,
            # 640.458333, 505.043478, 494.571429, 494.571429

            # With more engineered features (AutoViML)
            # 652.85, 640.458333, 640.458333, 640.458333,
            # 559.583333, 494.571429, 494.571429, 494.571429
            652.85, 645.0, 645.0, 645.0,
            640.458333, 505.043478, 485.444444, 485.444444
        ]
        # index = pd.RangeIndex(start=40, stop=48, step=1)
        # index = pd.to_datetime([
        index = [
            '2014-05-01', '2014-06-01', '2014-07-01', '2014-08-01',
            '2014-09-01', '2014-10-01', '2014-11-01', '2014-12-01'
            ]

        self.forecast_gold_ml_multivar_external_test_cv = pd.Series(
            data=results,
            index=index
        )
        self.forecast_gold_ml_multivar_external_test_cv.name = 'mean'
        self.forecast_gold_ml_multivar_external_test_cv.index.name = self.ts_column

        results = results[0:6]
        index = index[0:6]
        # self.forecast_gold_ml_multivar_external_test_10 = pd.Series(
        self.forecast_gold_ml_multivar_external_test_10_cv = pd.Series(
            data=results,
            index=index
        )
        # self.forecast_gold_ml_multivar_external_test_10.name = 'mean'
        self.forecast_gold_ml_multivar_external_test_10_cv.name = 'mean'
        self.forecast_gold_ml_multivar_external_test_10_cv.index.name = self.ts_column

    def test_auto_ts_multivar_ns_sarimax(self):
        """
        test to check functionality of the auto_ts function (multivariate with non seasonal SARIMAX)
        """

        print("\n\n" + "*"*50)
        print("Performing Unit Test: 'test_auto_ts_multivar_ns_SARIMAX'")
        print("*"*50 + "\n\n")

        # import numpy as np  # type: ignore
        # from auto_ts import auto_timeseries as ATS
        # seasonal_period argument does not make a difference since seasonality has been set to False.
        automl_model = ATS(
            score_type='rmse', forecast_period=self.forecast_period, time_interval='Month',
            # non_seasonal_pdq=None, seasonality=False, seasonal_period=12,
            # non_seasonal_pdq=None, seasonality=False, seasonal_period=3,
            non_seasonal_pdq=None, seasonality=False,
            model_type='best',
            verbose=0)
        automl_model.fit(
            traindata=self.train_multivar,
            ts_column=self.ts_column,
            target=self.target,
            cv=None # ,
            #sep=self.sep
            )
        _ = automl_model.predict(
            testdata=self.test_multivar[[self.ts_column] + self.preds],
            forecast_period=self.forecast_period
        )

        ml_dict = automl_model.get_ml_dict()

        leaderboard_gold = pd.DataFrame(
            {
                # 'name':['ML', 'Prophet', 'VAR', 'auto_SARIMAX', 'ARIMA'],
                'name':['ML', 'Prophet', 'VAR', 'auto_SARIMAX'],
                'rmse':[
                    self.rmse_gold_ml_multivar,
                    # self.rmse_gold_prophet_multivar_cv_fold2  # Unbiased
                    # FIXME: Biased as 2 validation periods even when CV is not mentioned
                    (self.rmse_gold_prophet_multivar_cv_fold2 + self.rmse_gold_prophet_multivar_cv_fold1)/2,
                    self.rmse_gold_var_multivar,
                    self.rmse_gold_auto_sarimax_multivar,
                    # self.rmse_gold_arima_uni_multivar
                ]
            }
        )
        assert_frame_equal(automl_model.get_leaderboard().reset_index(drop=True).round(6), leaderboard_gold)

        self.assertEqual(
            # automl_model.get_best_model_name(), "Prophet",  # Unbiased
            automl_model.get_best_model_name(), "ML",  # Biased
            "Best model name does not match expected value."
        )

        self.assertTrue(
            # isinstance(automl_model.get_best_model(), Prophet),  # Unbiased
            isinstance(automl_model.get_best_model(), BaseEstimator),  # Biased
            "Best model does not match expected value."
        )

        # TODO: Replace with Auto_ARIMA
        # self.assertTrue(
        #     isinstance(automl_model.get_model('SARIMAX'), SARIMAXResultsWrapper),
        #     "SARIMAX model does not match the expected type."
        # )


        ## Find a way to import these modules (BuildProphet & BuildSarimax) and then you can enable this.
        # self.assertTrue(
        #     isinstance(automl_model.get_best_model_build(), BuildProphet),
        #     "Best model build does not match expected value."
        # )

        # self.assertTrue(
        #     isinstance(automl_model.get_model_build('SARIMAX'), BuildSarimax),
        #     "SARIMAX model build does not match the expected type."
        # )


        # if automl_model.get_best_model_build() is not None:  # Unbiased
        if automl_model.get_model_build('Prophet') is not None:  # Biased (best model is no longer Prophet)
            # Simple forecast with forecast window = one used in training
            # Using default (best model)
            test_predictions = automl_model.predict(
                testdata=self.test_multivar[[self.ts_column] + self.preds],
                forecast_period=self.forecast_period,
                model="Prophet"  # Need to explicitly mention for Biased condirion since Prophet is no longer the best model
            )
            assert_series_equal(test_predictions['yhat'].round(6), self.forecast_gold_prophet_multivar_external_test_cv)

            # Simple forecast with forecast window != one used in training
            # Using default (best model)
            test_predictions = automl_model.predict(
                testdata=self.test_multivar[[self.ts_column] + self.preds],
                forecast_period=10,
                model="Prophet"  # Need to explicitly mention for Biased condirion since Prophet is no longer the best model
            )
            assert_series_equal(test_predictions['yhat'].round(6), self.forecast_gold_prophet_multivar_external_test_10_cv)

            # Simple forecast with forecast window = one used in training
            # Using named model
            test_predictions = automl_model.predict(
                testdata=self.test_multivar[[self.ts_column] + self.preds],
                forecast_period=self.forecast_period,
                model="Prophet"
            )
            assert_series_equal(test_predictions['yhat'].round(6), self.forecast_gold_prophet_multivar_external_test_cv)

            # Simple forecast with forecast window != one used in training
            # Using named model
            test_predictions = automl_model.predict(
                testdata=self.test_multivar[[self.ts_column] + self.preds],
                forecast_period=10,
                model="Prophet")
            assert_series_equal(test_predictions['yhat'].round(6), self.forecast_gold_prophet_multivar_external_test_10_cv)

            # Complex forecasts (returns confidence intervals, etc.)
            _ = automl_model.predict(
                testdata=self.test_multivar[[self.ts_column] + self.preds],
                forecast_period=self.forecast_period,
                model="Prophet",
                simple=False
            )
            # self.assertIsNone(
            #     np.testing.assert_array_equal(
            #         test_predictions.columns.values, self.expected_pred_col_names_prophet
            #     )
            # )


        # if automl_model.get_model_build('ARIMA') is not None:
        #     # Simple forecast with forecast window = one used in training
        #     # Using named model
        #     test_predictions = automl_model.predict(
        #         testdata=self.test_multivar[[self.ts_column] + self.preds], # Not needed for ARIMA
        #         forecast_period=self.forecast_period,
        #         model="ARIMA"
        #     )
        #     assert_series_equal(test_predictions['mean'].astype(np.double).round(6), self.forecast_gold_arima_uni_multivar_series)

        #     # Simple forecast with forecast window != one used in training
        #     # Using named model
        #     test_predictions = automl_model.predict(
        #         testdata=self.test_multivar[[self.ts_column] + self.preds], # Not needed for ARIMA
        #         forecast_period=10,
        #         model="ARIMA"
        #     )
        #     assert_series_equal(test_predictions['mean'].astype(np.double).round(6), self.forecast_gold_arima_uni_multivar_series_10)

        #     # Complex forecasts (returns confidence intervals, etc.)
        #     _ = automl_model.predict(
        #         testdata=self.test_multivar[[self.ts_column] + self.preds], # Not needed for ARIMA
        #         forecast_period=self.forecast_period,
        #         model="ARIMA",
        #         simple=False
        #     )
        #     # self.assertIsNone(
        #     #     np.testing.assert_array_equal(
        #     #         test_predictions.columns.values, self.expected_pred_col_names
        #     #     )
        #     # )

        # if automl_model.get_model_build('SARIMAX') is not None:
        #     # Simple forecast with forecast window = one used in training
        #     # Using named model
        #     test_predictions = automl_model.predict(
        #         forecast_period=self.forecast_period,
        #         testdata=self.test_multivar[[self.ts_column] + self.preds],
        #         model="SARIMAX"
        #     )

        #     assert_series_equal(test_predictions.round(6), self.forecast_gold_sarimax_multivar_external_test)

        #     # Simple forecast with forecast window != one used in training
        #     # Using named model
        #     test_predictions = automl_model.predict(
        #         forecast_period=6,
        #         testdata=self.test_multivar.iloc[0:6][[self.ts_column] + self.preds],
        #         model="SARIMAX"
        #     )
        #     assert_series_equal(test_predictions.round(6), self.forecast_gold_sarimax_multivar_external_test_10)

        #     # Complex forecasts (returns confidence intervals, etc.)
        #     test_predictions = automl_model.predict(
        #         forecast_period=self.forecast_period,
        #         testdata=self.test_multivar[[self.ts_column] + self.preds],
        #         model="SARIMAX",
        #         simple=False
        #     )
        #     self.assertIsNone(
        #         np.testing.assert_array_equal(
        #             test_predictions.columns.values, self.expected_pred_col_names
        #         )
        #     )

        #     # Checking missing exogenous variables
        #     with self.assertRaises(ValueError):
        #         _ = automl_model.predict(
        #             forecast_period=self.forecast_period,
        #             model="SARIMAX"
        #         )

        #     # Checking missing columns from exogenous variables
        #     with self.assertRaises(ValueError):
        #         test_multivar_temp = self.test_multivar.copy(deep=True)
        #         test_multivar_temp.rename(columns={'Marketing Expense': 'Incorrect Column'}, inplace=True)
        #         _ = automl_model.predict(
        #                 forecast_period=self.forecast_period,
        #                 testdata=test_multivar_temp,
        #                 model="SARIMAX"
        #             )

        #     test_predictions = automl_model.predict(
        #         forecast_period=10,
        #         testdata=self.test_multivar[[self.ts_column] + self.preds],
        #         model="SARIMAX"
        #     )
        #     assert_series_equal(test_predictions.round(6), self.forecast_gold_sarimax_multivar_external_test)

        if automl_model.get_model_build('ML') is not None:
            # Simple forecast with forecast window = one used in training
            # Using named model
            test_predictions = automl_model.predict(
                forecast_period=self.forecast_period,
                testdata=self.test_multivar[[self.ts_column] + self.preds],
                model="ML"
            )
            assert_series_equal(test_predictions['mean'].round(6), self.forecast_gold_ml_multivar_external_test)

            # Simple forecast with forecast window != one used in training
            # Using named model
            test_predictions = automl_model.predict(
                forecast_period=6,
                testdata=self.test_multivar.iloc[0:6][[self.ts_column] + self.preds],
                model="ML"
            )
            assert_series_equal(test_predictions['mean'].round(6), self.forecast_gold_ml_multivar_external_test_10)

            # Complex forecasts (returns confidence intervals, etc.)
            test_predictions = automl_model.predict(
                forecast_period=self.forecast_period,
                testdata=self.test_multivar[[self.ts_column] + self.preds],
                model="ML",
                simple=False
            )
            self.assertIsNone(
                np.testing.assert_array_equal(
                    test_predictions.columns.values, self.expected_pred_col_names
                )
            )

        ##################################
        #### Checking Prophet Results ####
        ##################################
        self.assertIsNone(
            np.testing.assert_array_equal(
                np.round(ml_dict.get('Prophet').get('forecast')[0]['yhat'], 6),
                # self.forecast_gold_prophet_multivar_internal_val_cv_fold2  # Unbiased
                self.forecast_gold_prophet_multivar_internal_val_cv_fold1  # Biased
            ),
            "(Multivar Test) Prophet Forecast does not match up with expected values."
        )

        self.assertEqual(
            # round(ml_dict.get('Prophet').get('rmse')[0], 8), self.rmse_gold_prophet_multivar_cv_fold2,  # Unbiased
            round(ml_dict.get('Prophet').get('rmse')[0], 8), self.rmse_gold_prophet_multivar_cv_fold1,  # Biased
            "(Multivar Test) Prophet RMSE does not match up with expected values.")

        # ################################
        # #### Checking ARIMA Results ####
        # ################################

        # # https://stackoverflow.com/questions/19387608/attributeerror-rint-when-using-numpy-round
        # self.assertIsNone(
        #     np.testing.assert_array_equal(
        #         np.round(ml_dict.get('ARIMA').get('forecast')['mean'].values.astype(np.double), 8),
        #         self.forecast_gold_arima_uni_multivar
        #     ),
        #     "(Multivar Test) ARIMA Forecast does not match up with expected values."
        # )

        # self.assertEqual(
        #     round(ml_dict.get('ARIMA').get('rmse'), 8), self.rmse_gold_arima_uni_multivar,
        #     "(Multivar Test) ARIMA RMSE does not match up with expected values.")

        # ##################################
        # #### Checking SARIMAX Results ####
        # ##################################
        # self.assertIsNone(
        #     np.testing.assert_array_equal(
        #         np.round(ml_dict.get('SARIMAX').get('forecast')[0]['mean'].values.astype(np.double), 6),
        #         self.forecast_gold_sarimax_multivar_internal_val
        #     ),
        #     "(Multivar Test) SARIMAX Forecast does not match up with expected values."
        # )

        # self.assertEqual(
        #     round(ml_dict.get('SARIMAX').get('rmse')[0], 6), self.rmse_gold_sarimax_multivar,
        #     "(Multivar Test) SARIMAX RMSE does not match up with expected values.")

        #############################
        #### Checking ML Results ####
        #############################

        self.assertListEqual(
            ml_dict.get('ML').get('forecast'), self.forecast_gold_ml_multivar_internal_val,
            "(Multivar Test) ML Forecast does not match up with expected values.")

        self.assertEqual(
            round(ml_dict.get('ML').get('rmse')[0], 6), self.rmse_gold_ml_multivar,
            "(Multivar Test) ML RMSE does not match up with expected values.")

    def test_auto_ts_univar_ns_sarimax(self):
        """
        test to check functionality of the auto_ts function (univariate models with non seasonal SARIMAX)
        """

        print("\n\n" + "*"*50)
        print("Performing Unit Test: 'test_auto_ts_univar_ns_SARIMAX'")
        print("*"*50 + "\n\n")

        # import numpy as np  # type: ignore
        # from auto_ts import auto_timeseries as ATS
        automl_model = ATS(
            score_type='rmse', forecast_period=self.forecast_period, time_interval='Month',
            non_seasonal_pdq=None, seasonality=False, seasonal_period=12,
            model_type='best',
            verbose=0)
        automl_model.fit(
            traindata=self.train_univar,
            ts_column=self.ts_column,
            target=self.target,
            cv=None # ,
            #sep=self.sep
            )
        _ = automl_model.predict(forecast_period=self.forecast_period)

        ml_dict = automl_model.get_ml_dict()

        leaderboard_gold = pd.DataFrame(
            {
                # 'name': ['Prophet', 'auto_SARIMAX', 'ARIMA', 'VAR', 'ML'],
                'name': ['Prophet', 'auto_SARIMAX', 'VAR', 'ML'],
                'rmse':[
                    self.rmse_gold_prophet_univar_cv_fold2,
                    self.rmse_gold_auto_sarimax_univar,
                    # self.rmse_gold_arima_uni_multivar,
                    math.inf,
                    math.inf
                ]
            }
        )
        assert_frame_equal(automl_model.get_leaderboard().reset_index(drop=True).round(6), leaderboard_gold)

        self.assertEqual(
            automl_model.get_best_model_name(), "Prophet",
            "Best model name does not match expected value."
        )
        self.assertTrue(
            isinstance(automl_model.get_best_model(), Prophet),
            "Best model does not match expected value."
        )

        ## Find a way to import these modules (BuildProphet & BuildSarimax) and then you can enable this.
        # self.assertTrue(
        #     isinstance(automl_model.get_best_model_build(), BuildProphet),
        #     "Best model build does not match expected value."
        # )

        # self.assertTrue(
        #     isinstance(automl_model.get_model_build('SARIMAX'), BuildSarimax),
        #     "SARIMAX model build does not match the expected type."
        # )

        if automl_model.get_best_model_build() is not None:
            # Simple forecast with forecast window = one used in training
            # Using default (best model)
            test_predictions = automl_model.predict(forecast_period=self.forecast_period)
            assert_series_equal(test_predictions['yhat'].round(6), self.forecast_gold_prophet_univar_external_test_cv)

            # Simple forecast with forecast window != one used in training
            # Using default (best model)
            test_predictions = automl_model.predict(forecast_period=10)
            assert_series_equal(test_predictions['yhat'].round(6), self.forecast_gold_prophet_univar_external_test_10_cv)

            # Simple forecast with forecast window = one used in training
            # Using named model
            test_predictions = automl_model.predict(
                forecast_period=self.forecast_period,
                model="Prophet"
            )
            assert_series_equal(test_predictions['yhat'].round(6), self.forecast_gold_prophet_univar_external_test_cv)

            # Simple forecast with forecast window != one used in training
            # Using named model
            test_predictions = automl_model.predict(
                forecast_period=10,
                model="Prophet")
            assert_series_equal(test_predictions['yhat'].round(6), self.forecast_gold_prophet_univar_external_test_10_cv)

            # Complex forecasts (returns confidence intervals, etc.)
            _ = automl_model.predict(
                forecast_period=self.forecast_period,
                model="Prophet",
                simple=False
            )
            # self.assertIsNone(
            #     np.testing.assert_array_equal(
            #         test_predictions.columns.values, self.expected_pred_col_names_prophet
            #     )
            # )


        # if automl_model.get_model_build('ARIMA') is not None:
        #     # Simple forecast with forecast window = one used in training
        #     # Using named model
        #     test_predictions = automl_model.predict(
        #         forecast_period=self.forecast_period,
        #         model="ARIMA"
        #     )
        #     assert_series_equal(test_predictions['mean'].astype(np.float).round(6), self.forecast_gold_arima_uni_multivar_series)

        #     # Simple forecast with forecast window != one used in training
        #     # Using named model
        #     test_predictions = automl_model.predict(
        #         forecast_period=10,
        #         model="ARIMA"
        #     )
        #     assert_series_equal(test_predictions['mean'].astype(np.float).round(6), self.forecast_gold_arima_uni_multivar_series_10)

        #     # Complex forecasts (returns confidence intervals, etc.)
        #     _ = automl_model.predict(
        #         forecast_period=self.forecast_period,
        #         model="ARIMA",
        #         simple=False
        #     )
        #     # self.assertIsNone(
        #     #     np.testing.assert_array_equal(
        #     #         test_predictions.columns.values, self.expected_pred_col_names
        #     #     )
        #     # )

        # # TODO: Change to auto_SARIMAX later
        # if automl_model.get_model_build('SARIMAX') is not None:
        #     # Simple forecast with forecast window = one used in training
        #     # Using named model
        #     test_predictions = automl_model.predict(
        #         forecast_period=self.forecast_period,
        #         model="SARIMAX"
        #     )
        #     assert_series_equal(test_predictions.round(6), self.forecast_gold_sarimax_univar_external_test)

        #     # Simple forecast with forecast window != one used in training
        #     # Using named model
        #     test_predictions = automl_model.predict(
        #         forecast_period=10,
        #         model="SARIMAX"
        #     )
        #     assert_series_equal(test_predictions.round(6), self.forecast_gold_sarimax_univar_external_test_10)

        #     # Complex forecasts (returns confidence intervals, etc.)
        #     test_predictions = automl_model.predict(
        #         forecast_period=self.forecast_period,
        #         model="SARIMAX",
        #         simple=False
        #     )
        #     self.assertIsNone(
        #         np.testing.assert_array_equal(
        #             test_predictions.columns.values, self.expected_pred_col_names
        #         )
        #     )
       

        if automl_model.get_model_build('ML') is not None:
            # Simple forecast with forecast window = one used in training
            # Using named model
            test_predictions = automl_model.predict(
                forecast_period=self.forecast_period,
                model="ML"
            )
            assert_series_equal(test_predictions['mean'].round(6), self.forecast_gold_ml_univar_series)

            # Simple forecast with forecast window != one used in training
            # Using named model
            test_predictions = automl_model.predict(
                forecast_period=6,
                model="ML"
            )
            assert_series_equal(test_predictions['mean'].round(6), self.forecast_gold_ml_univar_series_10)

            # Complex forecasts (returns confidence intervals, etc.)
            test_predictions = automl_model.predict(
                forecast_period=self.forecast_period,
                model="ML",
                simple=False
            )
            self.assertIsNone(
                np.testing.assert_array_equal(
                    test_predictions.columns.values, self.expected_pred_col_names
                )
            )



        #########################################
        #### Checking getter for Model Build ####
        #########################################
        self.assertIsNone(automl_model.get_model_build('ML'), "Expected Univar ML model to be None but did not get None.")


        ##################################
        #### Checking Prophet Results ####
        ##################################
        self.assertIsNone(
            np.testing.assert_array_equal(
                np.round(ml_dict.get('Prophet').get('forecast')[0]['yhat'], 6),
                # self.forecast_gold_prophet_univar_internal_val_cv_fold2  # Unbiased
                self.forecast_gold_prophet_univar_internal_val_cv_fold1  # Biased
            ),
            "(Univar Test) Prophet Forecast does not match up with expected values."
        )

        self.assertEqual(
            # round(ml_dict.get('Prophet').get('rmse')[0], 8), self.rmse_gold_prophet_univar_cv_fold2,  # Unbiased
            round(ml_dict.get('Prophet').get('rmse')[0], 8), self.rmse_gold_prophet_univar_cv_fold1,  # Biased
            "(Univar Test) Prophet RMSE does not match up with expected values.")

        # ################################
        # #### Checking ARIMA Results ####
        # ################################

        # # https://stackoverflow.com/questions/19387608/attributeerror-rint-when-using-numpy-round
        # self.assertIsNone(
        #     np.testing.assert_array_equal(
        #         np.round(ml_dict.get('ARIMA').get('forecast')['mean'].values.astype(np.double), 8),
        #         self.forecast_gold_arima_uni_multivar
        #     ),
        #     "(Univar Test) ARIMA Forecast does not match up with expected values."
        # )

        # self.assertEqual(
        #     round(ml_dict.get('ARIMA').get('rmse'), 8), self.rmse_gold_arima_uni_multivar,
        #     "(Univar Test) ARIMA RMSE does not match up with expected values.")

        # ##################################
        # #### Checking SARIMAX Results ####
        # ##################################
        # self.assertIsNone(
        #     np.testing.assert_array_equal(
        #         np.round(ml_dict.get('SARIMAX').get('forecast')[0]['mean'].values.astype(np.double), 8),
        #         self.forecast_gold_sarimax_univar_internal_val
        #     ),
        #     "(Univar Test) SARIMAX Forecast does not match up with expected values."
        # )

        # self.assertEqual(
        #     round(ml_dict.get('SARIMAX').get('rmse')[0],8), self.rmse_gold_sarimax_univar,
        #     "(Univar Test) SARIMAX RMSE does not match up with expected values.")

        #############################
        #### Checking ML Results ####
        #############################

        self.assertEqual(
            ml_dict.get('ML').get('forecast'), self.forecast_gold_ml_univar,
            "(Univar Test) ML Forecast does not match up with expected values."
        )

        self.assertEqual(
            round(ml_dict.get('ML').get('rmse'), 8), self.rmse_gold_ml_univar,
            "(Univar Test) ML RMSE does not match up with expected values."
        )

    def test_auto_ts_multivar_seasonal_sarimax(self):
        """
        test to check functionality of the auto_ts function (multivariate with seasonal SARIMAX)
        """

        print("\n\n" + "*"*50)
        print("Performing Unit Test: 'test_auto_ts_multivar_seasonal_SARIMAX'")
        print("*"*50 + "\n\n")

        # import numpy as np  # type: ignore
        # from auto_ts import auto_timeseries as ATS
        # TODO: seasonal_period argument does not make a difference. Commenting out for now.
        automl_model = ATS(
            score_type='rmse', forecast_period=self.forecast_period, time_interval='Month',
            non_seasonal_pdq=None, seasonality=True, seasonal_period=12,
            # non_seasonal_pdq=None, seasonality=True, seasonal_period=3,
            model_type=['auto_SARIMAX'],
            verbose=0)

        automl_model.fit(
            traindata=self.train_multivar,
            ts_column=self.ts_column,
            target=self.target,
            cv=None #,
            # sep=self.sep
            )

        _ = automl_model.predict(
            forecast_period=self.forecast_period,
            testdata=self.test_multivar[[self.ts_column] + self.preds],
        )

        ml_dict = automl_model.get_ml_dict()

        leaderboard_gold = pd.DataFrame(
            {
                'name':['auto_SARIMAX'],
                'rmse':[
                    self.rmse_gold_auto_sarimax_multivar_s12
                ]
            }
        )
        assert_frame_equal(automl_model.get_leaderboard().reset_index(drop=True).round(6), leaderboard_gold)

        # TODO: Replace with Auto_SARIMAX results
        # if automl_model.get_model_build('SARIMAX') is not None:
        #     # Simple forecast with forecast window = one used in training
        #     # Using named model
        #     test_predictions = automl_model.predict(
        #         forecast_period=self.forecast_period,
        #         testdata=self.test_multivar[[self.ts_column] + self.preds],
        #         model="SARIMAX"
        #     )

        #     assert_series_equal(test_predictions.round(6), self.forecast_gold_sarimax_multivar_external_test_s12)

        #     # Simple forecast with forecast window != one used in training
        #     # Using named model
        #     test_predictions = automl_model.predict(
        #         forecast_period=6,
        #         testdata=self.test_multivar.iloc[0:6][[self.ts_column] + self.preds],
        #         model="SARIMAX"
        #     )
        #     assert_series_equal(test_predictions.round(6), self.forecast_gold_sarimax_multivar_external_test_10_s12)

        #     # Complex forecasts (returns confidence intervals, etc.)
        #     test_predictions = automl_model.predict(
        #         forecast_period=self.forecast_period,
        #         testdata=self.test_multivar[[self.ts_column] + self.preds],
        #         model="SARIMAX",
        #         simple=False
        #     )
        #     self.assertIsNone(
        #         np.testing.assert_array_equal(
        #             test_predictions.columns.values, self.expected_pred_col_names
        #         )
        #     )

        #     # Checking missing exogenous variables
        #     with self.assertRaises(ValueError):
        #         test_predictions = automl_model.predict(
        #             forecast_period=self.forecast_period,
        #             # testdata=self.test_multivar[self.preds],
        #             model="SARIMAX"
        #         )

        #     # Checking missing columns from exogenous variables
        #     with self.assertRaises(ValueError):
        #         test_multivar_temp = self.test_multivar.copy(deep=True)
        #         test_multivar_temp.rename(columns={'Marketing Expense': 'Incorrect Column'}, inplace=True)
        #         test_predictions = automl_model.predict(
        #                 forecast_period=self.forecast_period,
        #                 testdata=test_multivar_temp,
        #                 model="SARIMAX"
        #             )

        #     test_predictions = automl_model.predict(
        #         forecast_period=10,
        #         testdata=self.test_multivar[[self.ts_column] + self.preds],
        #         model="SARIMAX"
        #     )
        #     assert_series_equal(test_predictions.round(6), self.forecast_gold_sarimax_multivar_external_test_s12)


        # ##################################
        # #### Checking SARIMAX Results ####
        # ##################################
        # self.assertIsNone(
        #     np.testing.assert_array_equal(
        #         np.round(ml_dict.get('SARIMAX').get('forecast')[0]['mean'].values.astype(np.double), 6),
        #         self.forecast_gold_sarimax_multivar_internal_val_s12
        #     ),
        #     "(Multivar Test) SARIMAX Forecast does not match up with expected values."
        # )

        # self.assertEqual(
        #     round(ml_dict.get('SARIMAX').get('rmse')[0], 6), self.rmse_gold_sarimax_multivar_s12,
        #     "(Multivar Test) SARIMAX RMSE does not match up with expected values.")

    def test_auto_ts_multivar_seasonal_sarimax_with_cv(self):
        """
        test to check functionality of the auto_ts function (multivariate with seasonal SARIMAX)
        """
        print("\n\n" + "*"*50)
        print("Performing Unit Test: 'test_auto_ts_multivar_seasonal_SARIMAX_withCV'")
        print("*"*50 + "\n\n")

        # import numpy as np  # type: ignore
        # from auto_ts import auto_timeseries as ATS

        # Reduced seasonality from 12 to 3 since we are doing CV. There is not enough data
        # to use seasonality of 12 along with CV since the first fold only has 24 train obs.
        automl_model = ATS(
            score_type='rmse', forecast_period=self.forecast_period, time_interval='Month',
            non_seasonal_pdq=None, seasonality=True, seasonal_period=3,
            model_type=['auto_SARIMAX'],
            verbose=0)

        automl_model.fit(
            traindata=self.train_multivar,
            ts_column=self.ts_column,
            target=self.target,
            cv=2 #,
            # sep=self.sep
            )

        _ = automl_model.predict(
            forecast_period=self.forecast_period,
            testdata=self.test_multivar[[self.ts_column] + self.preds],
        )

        _ = automl_model.get_ml_dict()

        leaderboard_gold = pd.DataFrame(
            {
                'name':['auto_SARIMAX'],
                'rmse':[
                    self.rmse_gold_auto_sarimax_multivar_s12_cv
                ]
            }
        )
        assert_frame_equal(automl_model.get_leaderboard().reset_index(drop=True).round(6), leaderboard_gold)

        # TODO: Replace with Auto_SARIMAX results
        # if automl_model.get_model_build('SARIMAX') is not None:
        #     # Simple forecast with forecast window = one used in training
        #     # Using named model
        #     test_predictions = automl_model.predict(
        #         forecast_period=self.forecast_period,
        #         testdata=self.test_multivar[[self.ts_column] + self.preds],
        #         model="SARIMAX"
        #     )

        #     assert_series_equal(test_predictions.round(6), self.forecast_gold_sarimax_multivar_external_test_s3_cv)

        #     # Simple forecast with forecast window != one used in training
        #     # Using named model
        #     test_predictions = automl_model.predict(
        #         forecast_period=6,
        #         testdata=self.test_multivar.iloc[0:6][[self.ts_column] + self.preds],
        #         model="SARIMAX"
        #     )
        #     assert_series_equal(test_predictions.round(6), self.forecast_gold_sarimax_multivar_external_test_10_s3_cv)

        #     # Complex forecasts (returns confidence intervals, etc.)
        #     test_predictions = automl_model.predict(
        #         forecast_period=self.forecast_period,
        #         testdata=self.test_multivar[[self.ts_column] + self.preds],
        #         model="SARIMAX",
        #         simple=False
        #     )
        #     self.assertIsNone(
        #         np.testing.assert_array_equal(
        #             test_predictions.columns.values, self.expected_pred_col_names
        #         )
        #     )

        #     # Checking missing exogenous variables
        #     with self.assertRaises(ValueError):
        #         test_predictions = automl_model.predict(
        #             forecast_period=self.forecast_period,
        #             # testdata=self.test_multivar[self.preds],
        #             model="SARIMAX"
        #         )

        #     # Checking missing columns from exogenous variables
        #     with self.assertRaises(ValueError):
        #         test_multivar_temp = self.test_multivar.copy(deep=True)
        #         test_multivar_temp.rename(columns={'Marketing Expense': 'Incorrect Column'}, inplace=True)
        #         test_predictions = automl_model.predict(
        #                 forecast_period=self.forecast_period,
        #                 testdata=test_multivar_temp,
        #                 model="SARIMAX"
        #             )

        #     test_predictions = automl_model.predict(
        #         forecast_period=10,
        #         testdata=self.test_multivar[[self.ts_column] + self.preds],
        #         model="SARIMAX"
        #     )
        #     assert_series_equal(test_predictions.round(6), self.forecast_gold_sarimax_multivar_external_test_s3_cv)


        # ##################################
        # #### Checking SARIMAX Results ####
        # ##################################
        # self.assertIsNone(
        #     np.testing.assert_array_equal(
        #         np.round(ml_dict.get('SARIMAX').get('forecast')[0]['mean'].values.astype(np.double), 6),
        #         self.forecast_gold_sarimax_multivar_internal_val_s3_cv_fold1
        #     ),
        #     "(Multivar Test) SARIMAX Forecast does not match up with expected values --> Fold 1."
        # )
        # self.assertIsNone(
        #     np.testing.assert_array_equal(
        #         np.round(ml_dict.get('SARIMAX').get('forecast')[1]['mean'].values.astype(np.double), 6),
        #         self.forecast_gold_sarimax_multivar_internal_val_s3_cv_fold2
        #     ),
        #     "(Multivar Test) SARIMAX Forecast does not match up with expected values --> Fold 2."
        # )

        # self.assertEqual(
        #     round(ml_dict.get('SARIMAX').get('rmse')[0], 6), self.rmse_gold_sarimax_multivar_s3_cv_fold1,
        #     "(Multivar Test) SARIMAX RMSE does not match up with expected values --> Fold 1.")
        # self.assertEqual(
        #     round(ml_dict.get('SARIMAX').get('rmse')[1], 6), self.rmse_gold_sarimax_multivar_s3_cv_fold2,
        #     "(Multivar Test) SARIMAX RMSE does not match up with expected values --> Fold 2.")

    def test_subset_of_models(self):
        """
        test to check functionality of the training with only a subset of models
        """

        print("\n\n" + "*"*50)
        print("Performing Unit Test: 'test_subset_of_models'")
        print("*"*50 + "\n\n")

        # import numpy as np  # type: ignore
        # from auto_ts import auto_timeseries as ATS

        automl_model = ATS(
            score_type='rmse', forecast_period=self.forecast_period, time_interval='Month',
            non_seasonal_pdq=None, seasonality=False, seasonal_period=12,
            model_type=['SARIMAX', 'auto_SARIMAX', 'ML'],
            verbose=0)
        automl_model.fit(
            traindata=self.train_multivar,
            ts_column=self.ts_column,
            target=self.target,
            cv=None,
            sep=self.sep
        )

        leaderboard_gold = pd.DataFrame(
            {
                'name': ['ML', 'auto_SARIMAX'],
                'rmse': [
                    self.rmse_gold_ml_multivar,
                    self.rmse_gold_auto_sarimax_multivar
                ]
            }
        )
        assert_frame_equal(automl_model.get_leaderboard().reset_index(drop=True).round(6), leaderboard_gold)

        automl_model = ATS(
            score_type='rmse', forecast_period=self.forecast_period, time_interval='Month',
            non_seasonal_pdq=None, seasonality=False, seasonal_period=12,
            model_type=['auto_SARIMAX', 'bogus', 'ML'],
            verbose=0)
        automl_model.fit(
            traindata=self.train_multivar,
            ts_column=self.ts_column,
            target=self.target,
            cv=None,
            sep=self.sep)

        leaderboard_gold = pd.DataFrame(
            {
                'name': ['ML', 'auto_SARIMAX'],
                'rmse': [self.rmse_gold_ml_multivar, self.rmse_gold_auto_sarimax_multivar]
            }
        )
        assert_frame_equal(automl_model.get_leaderboard().reset_index(drop=True).round(6), leaderboard_gold)

        automl_model = ATS(
            score_type='rmse', forecast_period=self.forecast_period, time_interval='Month',
            non_seasonal_pdq=None, seasonality=False, seasonal_period=12,
            model_type=['bogus'],
            verbose=0)
        status = automl_model.fit(
            traindata=self.train_multivar,
            ts_column=self.ts_column,
            target=self.target,
            cv=None,
            sep=self.sep)
        self.assertIsNone(status)

    def test_passing_list_instead_of_str(self):
        """
        Tests passing models as a list instead of a string
        """

        print("\n\n" + "*"*50)
        print("Performing Unit Test: 'test_passing_list_instead_of_str'")
        print("*"*50 + "\n\n")

        # import numpy as np  # type: ignore
        # from auto_ts import auto_timeseries as ATS

        automl_model = ATS(
            score_type='rmse', forecast_period=self.forecast_period, time_interval='Month',
            non_seasonal_pdq=None, seasonality=False, seasonal_period=12,
            model_type=['auto_SARIMAX', 'ML'],
            verbose=0)
        automl_model.fit(
            traindata=self.train_multivar,
            ts_column=[self.ts_column],
            target=[self.target],
            cv=None,
            sep=self.sep)

        leaderboard_models = np.array(['ML', 'auto_SARIMAX'])
        np.testing.assert_array_equal(automl_model.get_leaderboard()['name'].values, leaderboard_models)

    def test_cv_retrieval_plotting(self):
        """
        Tests CV Scores retrieval and plotting
        """

        print("\n\n" + "*"*50)
        print("Performing Unit Test: 'test_cv_retrieval_plotting'")
        print("*"*50 + "\n\n")

        # import numpy as np  # type: ignore
        # from auto_ts import auto_timeseries as ATS


        automl_model = ATS(
            score_type='rmse', forecast_period=self.forecast_period, time_interval='Month',
            non_seasonal_pdq=None, seasonality=False, seasonal_period=12,
            model_type=['ML', 'auto_SARIMAX'],
            verbose=0)
        automl_model.fit(
            traindata=self.train_multivar,
            ts_column=self.ts_column,
            target=self.target,
            cv=2,
            sep=self.sep)

        cv_scores_gold = pd.DataFrame(
            {
                'Model': ['auto_SARIMAX', 'auto_SARIMAX', 'ML', 'ML'],
                # 'CV Scores': [73.2824, 185.705, 93.0305, 67.304]
                'CV Scores': [100.816639, 147.703077, 107.2037, 81.7406]  # With more engineered features (AutoViML)
            }
        )
        cv_scores = automl_model.get_cv_scores()
        assert_frame_equal(cv_scores, cv_scores_gold)

        automl_model.plot_cv_scores()

    def test_prophet_multivar_standalone_no_cv(self):
        """
        test to check functionality Prophet with CV
        """
        print("\n\n" + "*"*50)
        print("Performing Unit Test: 'test_prophet_multivar_standalone_noCV'")
        print("*"*50 + "\n\n")

        # import numpy as np  # type: ignore
        # from auto_ts import auto_timeseries as ATS

        automl_model = ATS(
            score_type='rmse', forecast_period=self.forecast_period, time_interval='Month',
            non_seasonal_pdq=None, seasonality=False, seasonal_period=12,
            model_type=['prophet'],
            verbose=0)
        automl_model.fit(
            traindata=self.train_multivar,
            ts_column=self.ts_column,
            target=self.target,
            cv=None,
            sep=self.sep)

        ml_dict = automl_model.get_ml_dict()

        ####################################
        #### Internal Validation Checks ####
        ####################################

        ## Validation Forecast
        self.assertIsNone(
            np.testing.assert_array_equal(
                np.round(ml_dict.get('Prophet').get('forecast')[0]['yhat'], 6),
                # self.forecast_gold_prophet_multivar_internal_val_cv_fold2  # Unbiased
                self.forecast_gold_prophet_multivar_internal_val_cv_fold1    # Biased
            )
        )

        # Validation RMSE
        self.assertEqual(
            # round(ml_dict.get('Prophet').get('rmse')[0], 8), self.rmse_gold_prophet_multivar_cv_fold2  # Unbiased
            round(ml_dict.get('Prophet').get('rmse')[0], 8), self.rmse_gold_prophet_multivar_cv_fold1  # Biased
        )

        ##############################
        #### External Test Checks ####
        ##############################

        # Simple forecast with forecast window = one used in training
        # Using named model
        test_predictions = automl_model.predict(
            testdata=self.test_multivar[[self.ts_column] + self.preds],
            forecast_period=self.forecast_period,
            model="Prophet"
        )

        assert_series_equal(test_predictions['yhat'].round(6), self.forecast_gold_prophet_multivar_external_test_cv)

        # Simple forecast with forecast window != one used in training
        # Using named model
        test_predictions = automl_model.predict(
            testdata=self.test_multivar[[self.ts_column] + self.preds],
            forecast_period=10,
            model="Prophet")
        assert_series_equal(test_predictions['yhat'].round(6), self.forecast_gold_prophet_multivar_external_test_10_cv)

    def test_prophet_multivar_standalone_with_cv(self):
        """
        test to check functionality Prophet with CV
        """
        print("\n\n" + "*"*50)
        print("Performing Unit Test: 'test_prophet_multivar_standalone_withCV'")
        print("*"*50 + "\n\n")

        # import numpy as np  # type: ignore
        # from auto_ts import auto_timeseries as ATS

        automl_model = ATS(
            score_type='rmse', forecast_period=self.forecast_period, time_interval='Month',
            non_seasonal_pdq=None, seasonality=False, seasonal_period=12,
            model_type=['prophet'],
            verbose=0)
        automl_model.fit(
            traindata=self.train_multivar,
            ts_column=self.ts_column,
            target=self.target,
            cv=2,
            sep=self.sep)

        ml_dict = automl_model.get_ml_dict()

        ####################################
        #### Internal Validation Checks ####
        ####################################

        ## Validation Forecast
        self.assertIsNone(
            np.testing.assert_array_equal(
                np.round(ml_dict.get('Prophet').get('forecast')[0]['yhat'], 6),
                self.forecast_gold_prophet_multivar_internal_val_cv_fold1
            )
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                np.round(ml_dict.get('Prophet').get('forecast')[1]['yhat'], 6),
                self.forecast_gold_prophet_multivar_internal_val_cv_fold2
            )
        )

        # Validation RMSE
        self.assertEqual(
            round(ml_dict.get('Prophet').get('rmse')[0], 8), self.rmse_gold_prophet_multivar_cv_fold1
        )
        self.assertEqual(
            round(ml_dict.get('Prophet').get('rmse')[1], 8), self.rmse_gold_prophet_multivar_cv_fold2
        )

        ##############################
        #### External Test Checks ####
        ##############################

        # Simple forecast with forecast window = one used in training
        # Using named model
        test_predictions = automl_model.predict(
            testdata=self.test_multivar[[self.ts_column] + self.preds],
            forecast_period=self.forecast_period,
            model="Prophet"
        )
        assert_series_equal(test_predictions['yhat'].round(6), self.forecast_gold_prophet_multivar_external_test_cv)

        # Simple forecast with forecast window != one used in training
        # Using named model
        test_predictions = automl_model.predict(
            testdata=self.test_multivar[[self.ts_column] + self.preds],
            forecast_period=10,
            model="Prophet")
        assert_series_equal(test_predictions['yhat'].round(6), self.forecast_gold_prophet_multivar_external_test_10_cv)

    def test_ml_standalone_no_cv(self):
        """
        Testing ML Standalone without CV
        """

        print("\n\n" + "*"*50)
        print("Performing Unit Test: 'test_ml_standalone_noCV'")
        print("*"*50 + "\n\n")

        # import numpy as np  # type: ignore
        # from auto_ts import auto_timeseries as ATS

        automl_model = ATS(
            score_type='rmse', forecast_period=self.forecast_period, time_interval='Month',
            non_seasonal_pdq=None, seasonality=False, seasonal_period=12,
            model_type=['ML'],
            verbose=0)
        automl_model.fit(
            traindata=self.train_multivar,
            ts_column=self.ts_column,
            target=self.target,
            cv=None,
            sep=self.sep)


    def test_ml_standalone_with_cv(self):
        """
        Testing ML Standalone with CV
        """
        print("\n\n" + "*"*50)
        print("Performing Unit Test: 'test_ml_standalone_withCV'")
        print("*"*50 + "\n\n")

        # import numpy as np  # type: ignore
        # from auto_ts import auto_timeseries as ATS

        automl_model = ATS(
            score_type='rmse', forecast_period=self.forecast_period, time_interval='Month',
            non_seasonal_pdq=None, seasonality=False, seasonal_period=12,
            model_type=['ML'],
            verbose=0)
        automl_model.fit(
            traindata=self.train_multivar,
            ts_column=self.ts_column,
            target=self.target,
            cv=2,
            sep=self.sep)

        ml_dict = automl_model.get_ml_dict()

        leaderboard_gold = pd.DataFrame(
            {
                'name':['ML'],
                'rmse':[self.rmse_gold_ml_multivar_cv]
            }
        )
        assert_frame_equal(automl_model.get_leaderboard().reset_index(drop=True).round(6), leaderboard_gold)

        if automl_model.get_model_build('ML') is not None:
            # Simple forecast with forecast window = one used in training
            # Using named model
            test_predictions = automl_model.predict(
                forecast_period=self.forecast_period,
                testdata=self.test_multivar[[self.ts_column] + self.preds],
                model="ML"
            )
            assert_series_equal(test_predictions['mean'].round(6), self.forecast_gold_ml_multivar_external_test_cv)

            # Simple forecast with forecast window != one used in training
            # Using named model
            test_predictions = automl_model.predict(
                forecast_period=6,
                testdata=self.test_multivar.iloc[0:6][[self.ts_column] + self.preds],
                model="ML"
            )
            assert_series_equal(test_predictions['mean'].round(6), self.forecast_gold_ml_multivar_external_test_10_cv)

            # Complex forecasts (returns confidence intervals, etc.)
            test_predictions = automl_model.predict(
                forecast_period=self.forecast_period,
                testdata=self.test_multivar[[self.ts_column] + self.preds],
                model="ML",
                simple=False
            )
            self.assertIsNone(
                np.testing.assert_array_equal(
                    test_predictions.columns.values, self.expected_pred_col_names
                )
            )

            # For ML forecast window is ignored (horizon only depends on the length of testdata)
            # Hence even though we specify forecast_period of 10, it only takes 8 here
            test_predictions = automl_model.predict(
                forecast_period=10,
                testdata=self.test_multivar[[self.ts_column] + self.preds],
                model="ML"
            )
            assert_series_equal(test_predictions['mean'].round(6), self.forecast_gold_ml_multivar_external_test_cv)

        # RMSE check for each fold
        self.assertEqual(
            round(ml_dict.get('ML').get('rmse')[0], 8), self.rmse_gold_ml_multivar_cv_fold1,
            "(Multivar Test) ML RMSE does not match up with expected values --> Fold 1.")
        self.assertEqual(
            round(ml_dict.get('ML').get('rmse')[1], 8), self.rmse_gold_ml_multivar_cv_fold2,
            "(Multivar Test) ML RMSE does not match up with expected values --> Fold 2.")


if __name__ == '__main__':
    unittest.main()
