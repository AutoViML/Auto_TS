import unittest
import math
import numpy as np # type: ignore
import pandas as pd # type: ignore

from pandas.testing import assert_series_equal # type: ignore
from pandas.testing import assert_frame_equal # type: ignore
from fbprophet.forecaster import Prophet # type: ignore
from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper  # type: ignore


class TestAutoTS(unittest.TestCase):

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
            447.733878, 510.152411, 536.119254, 599.663702,
            592.020981, 657.844486, 667.245238, 485.989273
        ])

        self.forecast_gold_prophet_univar_internal_val_cv_fold2 = np.array([            
            614.482071, 562.050462, 534.810663, 605.566298,  
            580.899233, 585.676464, 686.480721, 732.167184
        ])

        self.rmse_gold_prophet_univar_cv_fold1 = 86.34827037
        self.rmse_gold_prophet_univar_cv_fold2 = 56.5751 # Without CV gets this result
        

        ## External Test Set results 
        results = [
            749.061242, 751.077262, 796.892366, 783.206733,
            689.698130, 595.713426, 569.486600, 635.884371           
            ]
        index = np.arange(40, 48)

        self.forecast_gold_prophet_univar_external_test_cv = pd.Series(
                data = results,
                index = index
            )
        self.forecast_gold_prophet_univar_external_test_cv.name = 'yhat'

        results = results + [576.473786, 581.275889]
        index = np.arange(40, 50)
            
        self.forecast_gold_prophet_univar_external_test_10_cv = pd.Series(
                data = results, 
                index = index
            )
        self.forecast_gold_prophet_univar_external_test_10_cv.name = 'yhat'

        
        ###########################################################
        #### MULTIVARIATE [Both No CV (uses fold2) and CV = 2] ####
        ###########################################################
        
        # Internal (to AutoML) validation set results
        self.forecast_gold_prophet_multivar_internal_val_cv_fold1 = np.array([            
            502.111972, 569.181958, 578.128706, 576.069791,
            663.258686, 677.851419, 750.972617, 781.269791
        ])

        self.forecast_gold_prophet_multivar_internal_val_cv_fold2 = np.array([            
            618.244315, 555.784628, 524.396122, 611.513751,
            584.936717, 605.940656, 702.652641, 736.639273
        ])

        self.rmse_gold_prophet_multivar_cv_fold1 = 48.70419901 
        self.rmse_gold_prophet_multivar_cv_fold2 = 63.24631835 # Without CV gets this result 
        
        
        ## External Test Set results 
        results = [
            747.964093, 736.512241, 814.840792, 825.152970,
            657.743450, 588.985816, 556.814528, 627.768202  
            ]
        index = np.arange(0, 8)

        self.forecast_gold_prophet_multivar_external_test_cv = pd.Series(
                data = results,
                index = index
            )
        self.forecast_gold_prophet_multivar_external_test_cv.name = 'yhat'

        # Same as regular since we only have 8 exogenous observations
        self.forecast_gold_prophet_multivar_external_test_10_cv = pd.Series(
                data = results, 
                index = index
            )
        self.forecast_gold_prophet_multivar_external_test_10_cv.name = 'yhat'

        
        ##############################
        #### ARIMA Golden Results ####
        ##############################

        #### UNIVARIATE and MULTIVARIATE ####

        results = [
            801.78660584, 743.16044526, 694.38764549, 684.72931967,
            686.70229610, 692.13402266, 698.59426282, 705.36034762            
            ]
        index = [
            'Forecast_1', 'Forecast_2', 'Forecast_3', 'Forecast_4',
            'Forecast_5', 'Forecast_6', 'Forecast_7', 'Forecast_8'  
            ]

        self.forecast_gold_arima_uni_multivar = np.array(results)

        self.forecast_gold_arima_uni_multivar_series = pd.Series(
            data = results,
            index = index
        )
        self.forecast_gold_arima_uni_multivar_series.name = 'mean'

        results = results + [712.217380, 719.101457]
        index = index + ['Forecast_9', 'Forecast_10']

        self.forecast_gold_arima_uni_multivar_series_10 = pd.Series(
            data = results,
            index = index
        )
        self.forecast_gold_arima_uni_multivar_series_10.name = 'mean'

        self.rmse_gold_arima_uni_multivar = 169.00016628
        

        #######################################################################################################

        ################################
        #### SARIMAX Golden Results ####
        ################################

        #### UNIVARIATE ####

        ## Internal (to AutoML) validation set results
        results = [
            803.31673726, 762.46093997, 718.3581931,  711.42130506,
            719.36254603, 732.70981867, 747.57645435, 762.47349398            
            ]
        self.forecast_gold_sarimax_univar_internal_val = np.array(results)
        self.rmse_gold_sarimax_univar = 193.49650578

        ## External Test Set results
        results=[
            737.281499, 718.144765, 672.007487, 618.321458,
            578.990868, 567.799468, 586.467414, 625.619993
        ]
        index = pd.to_datetime([
            '2014-05-01', '2014-06-01', '2014-07-01', '2014-08-01',
            '2014-09-01', '2014-10-01', '2014-11-01', '2014-12-01'
            ])
        
        self.forecast_gold_sarimax_univar_external_test = pd.Series(
                data = results,
                index = index
            )
        self.forecast_gold_sarimax_univar_external_test.name = 'mean'

        results = results + [669.666326, 703.29552]
        index = pd.to_datetime([
            '2014-05-01', '2014-06-01', '2014-07-01', '2014-08-01',
            '2014-09-01', '2014-10-01', '2014-11-01', '2014-12-01',
            '2015-01-01', '2015-02-01'
            ])
            
        self.forecast_gold_sarimax_univar_external_test_10 = pd.Series(
                data = results,
                index = index
            )
        self.forecast_gold_sarimax_univar_external_test_10.name = 'mean'

        
        #######################################
        #### MULTIVARIATE (no seasonality) ####
        #######################################
                        
        ## Internal (to AutoML) validation set results
        results = [
            772.268886, 716.337431, 686.167231, 739.269047,
            704.280567, 757.450733, 767.711055, 785.960125
        ]
        self.forecast_gold_sarimax_multivar_internal_val = np.array(results)
        self.rmse_gold_sarimax_multivar = 185.704684  

        ## External Test Set results (With Multivariate columns accepted)
        results = [
            750.135204, 806.821297, 780.232195, 743.309074,
            724.400616, 683.117893, 673.696113, 686.807075
        ]
        index = pd.to_datetime([
            '2014-05-01', '2014-06-01', '2014-07-01', '2014-08-01',
            '2014-09-01', '2014-10-01', '2014-11-01', '2014-12-01'
            ])
        
        self.forecast_gold_sarimax_multivar_external_test = pd.Series(
                data = results,
                index = index
            )
        self.forecast_gold_sarimax_multivar_external_test.name = 'mean'

        results = results[0:6] 
        index = index[0:6]
        self.forecast_gold_sarimax_multivar_external_test_10 = pd.Series(
                data = results,
                index = index
            )
        self.forecast_gold_sarimax_multivar_external_test_10.name = 'mean'

        ################################################################################
        #### MULTIVARIATE (with seasonality = True, Seasonal Period = 12 CV = None) ####
        ################################################################################

        ## Internal (to AutoML) validation set results (with seasonality = True, Seasonal Period = 12)
        # Without CV
        results = [
            726.115602, 646.028979, 657.249936, 746.752393,
            732.813245, 749.435178, 863.356789, 903.168728 
        ]
        self.forecast_gold_sarimax_multivar_internal_val_s12 = np.array(results)
        self.rmse_gold_sarimax_multivar_s12 = 197.18894

        ## External Test Set results (With Multivariate columns accepted) (with seasonality = True, Seasonal Period = 12)
        
        results = [
            1006.134134, 779.874076, 420.461804, 724.042104,
            1827.304601, 1204.070838, -2216.439611, -1278.974132
        ]

        index = pd.to_datetime([
            '2014-05-01', '2014-06-01', '2014-07-01', '2014-08-01',
            '2014-09-01', '2014-10-01', '2014-11-01', '2014-12-01'
            ])
                
        self.forecast_gold_sarimax_multivar_external_test_s12 = pd.Series(
                data = results,
                index = index
            )
        self.forecast_gold_sarimax_multivar_external_test_s12.name = 'mean'

        results = results[0:6] 
        index = index[0:6]
        self.forecast_gold_sarimax_multivar_external_test_10_s12 = pd.Series(
                data = results,
                index = index
            )
        self.forecast_gold_sarimax_multivar_external_test_10_s12.name = 'mean'

        #############################################################################
        #### MULTIVARIATE (with seasonality = True, Seasonal Period = 3, CV = 2) ####
        #############################################################################

        ## Internal (to AutoML) validation set results 
        
        results = [
            119.260686, 540.623654, 230.040446, 364.088969,
            470.581971, 105.559723, 84.335069, 110.757574 
        ]
        self.forecast_gold_sarimax_multivar_internal_val_s3_cv_fold1 = np.array(results)

        results = [
            551.736392, 502.232401, 440.047123, 521.382176,
            496.012325, 501.563083, 634.825011, 674.975611
        ]
        self.forecast_gold_sarimax_multivar_internal_val_s3_cv_fold2 = np.array(results)

        self.rmse_gold_sarimax_multivar_s12_cv = 239.191102
        self.rmse_gold_sarimax_multivar_s3_cv_fold1 = 443.839435
        self.rmse_gold_sarimax_multivar_s3_cv_fold2 = 34.542769


        ## External Test Set results 
        results = [
            770.447134, 784.881945, 857.496478, 918.626627,
            689.107408, 599.827292, 608.747367, 634.957579
        ]

        index = pd.to_datetime([
            '2014-05-01', '2014-06-01', '2014-07-01', '2014-08-01',
            '2014-09-01', '2014-10-01', '2014-11-01', '2014-12-01'
            ])
                
        self.forecast_gold_sarimax_multivar_external_test_s3_cv = pd.Series(
                data = results,
                index = index
            )
        self.forecast_gold_sarimax_multivar_external_test_s3_cv.name = 'mean'

        results = results[0:6] 
        index = index[0:6]
        self.forecast_gold_sarimax_multivar_external_test_10_s3_cv = pd.Series(
                data = results,
                index = index
            )
        self.forecast_gold_sarimax_multivar_external_test_10_s3_cv.name = 'mean'


        #######################################################################################################

        ############################
        #### VAR Golden Results ####
        ############################

        #### UNIVARIATE ####
        self.forecast_gold_var_univar = None
        self.rmse_gold_var_univar = math.inf
        self.forecast_gold_var_univar_series = None
        self.forecast_gold_var_univar_series_10 = None

        #### MULTIVARIATE ####
        results = [
            741.37790864, 676.23341949, 615.53872102, 571.7977285,
            546.95278336, 537.34223069, 537.4744872,  542.30739271           
            ]

        index = pd.to_datetime([
            '2013-09-01', '2013-10-01', '2013-11-01', '2013-12-01',
            '2014-01-01', '2014-02-01', '2014-03-01', '2014-04-01'
            ])

        self.forecast_gold_var_multivar = np.array(results)

        self.forecast_gold_var_multivar_series = pd.Series(
                data = results,
                index = index
            )
        self.forecast_gold_var_multivar_series.name = 'mean'

        results = results + [548.245948, 553.274306]
        index = pd.to_datetime([
            '2013-09-01', '2013-10-01', '2013-11-01', '2013-12-01',
            '2014-01-01', '2014-02-01', '2014-03-01', '2014-04-01',
            '2014-05-01', '2014-06-01'
            ])
            
        self.forecast_gold_var_multivar_series_10 = pd.Series(
                data = results,
                index = index
            )
        self.forecast_gold_var_multivar_series_10.name = 'mean'

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
            733.293931, 627.633457, 621.182141, 614.128809,
            600.902623, 451.565462, 330.694427, 348.744604            
        ]
        index = pd.RangeIndex(start=40, stop=48, step=1) 

        self.forecast_gold_ml_multivar_external_test = pd.Series(
                data = results,
                index = index
            )
        self.forecast_gold_ml_multivar_external_test.name = 'mean'

        results = results[0:6] 
        index = index[0:6]            
        #self.forecast_gold_ml_multivar_external_test_10_cv = pd.Series(
        self.forecast_gold_ml_multivar_external_test_10 = pd.Series(
                data = results,
                index = index
            )
        # self.forecast_gold_ml_multivar_external_test_10_cv.name = 'mean'
        self.forecast_gold_ml_multivar_external_test_10.name = 'mean'

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
            652.85, 640.458333, 640.458333, 640.458333,
            559.583333, 494.571429, 494.571429, 494.571429
        ]
        index = pd.RangeIndex(start=40, stop=48, step=1) 

        self.forecast_gold_ml_multivar_external_test_cv = pd.Series(
                data = results,
                index = index
            )
        self.forecast_gold_ml_multivar_external_test_cv.name = 'mean'

        results = results[0:6] 
        index = index[0:6]            
        # self.forecast_gold_ml_multivar_external_test_10 = pd.Series(
        self.forecast_gold_ml_multivar_external_test_10_cv = pd.Series(
                data = results,
                index = index
            )
        # self.forecast_gold_ml_multivar_external_test_10.name = 'mean'
        self.forecast_gold_ml_multivar_external_test_10_cv.name = 'mean'

    # @unittest.skip
    def test_auto_ts_multivar_ns_SARIMAX(self):
        """
        test to check functionality of the auto_ts function (multivariate with non seasonal SARIMAX)
        """

        print("\n\n" + "*"*50)
        print("Performing Unit Test: 'test_auto_ts_multivar_ns_SARIMAX'")
        print("*"*50 + "\n\n")

        import numpy as np  # type: ignore
        from auto_ts import AutoTimeSeries as ATS
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
        test_predictions = automl_model.predict(
            X_exogen=self.test_multivar[[self.ts_column] + self.preds], 
            forecast_period=self.forecast_period
        )  
        
        ml_dict = automl_model.get_ml_dict()
        
        leaderboard_gold = pd.DataFrame(
            {
                'name':['Prophet', 'ML', 'VAR', 'ARIMA', 'SARIMAX', 'PyFlux'],
                'rmse':[
                    self.rmse_gold_prophet_multivar_cv_fold2,
                    self.rmse_gold_ml_multivar,
                    self.rmse_gold_var_multivar,
                    self.rmse_gold_arima_uni_multivar,
                    self.rmse_gold_sarimax_multivar,
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
        
        self.assertTrue(
            isinstance(automl_model.get_model('SARIMAX'), SARIMAXResultsWrapper),
            "SARIMAX model does not match the expected type."
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
            test_predictions = automl_model.predict(
                X_exogen=self.test_multivar[[self.ts_column] + self.preds],
                forecast_period=self.forecast_period                    
            )
            assert_series_equal(test_predictions.round(6), self.forecast_gold_prophet_multivar_external_test_cv)        

            # Simple forecast with forecast window != one used in training
            # Using default (best model)
            test_predictions = automl_model.predict(
                X_exogen=self.test_multivar[[self.ts_column] + self.preds],
                forecast_period=10
            )
            assert_series_equal(test_predictions.round(6), self.forecast_gold_prophet_multivar_external_test_10_cv)   

            # Simple forecast with forecast window = one used in training
            # Using named model
            test_predictions = automl_model.predict(
                X_exogen=self.test_multivar[[self.ts_column] + self.preds],
                forecast_period=self.forecast_period,
                model="Prophet"
            )
            assert_series_equal(test_predictions.round(6), self.forecast_gold_prophet_multivar_external_test_cv)        

            # Simple forecast with forecast window != one used in training
            # Using named model
            test_predictions = automl_model.predict(
                X_exogen=self.test_multivar[[self.ts_column] + self.preds],
                forecast_period=10,
                model="Prophet")
            assert_series_equal(test_predictions.round(6), self.forecast_gold_prophet_multivar_external_test_10_cv)        

            # Complex forecasts (returns confidence intervals, etc.)
            test_predictions = automl_model.predict(
                X_exogen=self.test_multivar[[self.ts_column] + self.preds],
                forecast_period=self.forecast_period,
                model="Prophet",
                simple=False
            )
            # print(test_predictions)
            # self.assertIsNone(
            #     np.testing.assert_array_equal(
            #         test_predictions.columns.values, self.expected_pred_col_names_prophet
            #     )
            # )
        

        if automl_model.get_model_build('ARIMA') is not None:
            # Simple forecast with forecast window = one used in training
            # Using named model
            test_predictions = automl_model.predict(
                X_exogen=self.test_multivar[[self.ts_column] + self.preds], # Not needed for ARIMA
                forecast_period=self.forecast_period,
                model="ARIMA"
            )
            assert_series_equal(test_predictions.round(6), self.forecast_gold_arima_uni_multivar_series) 
            
            # Simple forecast with forecast window != one used in training
            # Using named model
            test_predictions = automl_model.predict(
                X_exogen=self.test_multivar[[self.ts_column] + self.preds], # Not needed for ARIMA
                forecast_period=10,
                model="ARIMA"
            )
            assert_series_equal(test_predictions.round(6), self.forecast_gold_arima_uni_multivar_series_10) 

            # Complex forecasts (returns confidence intervals, etc.)
            test_predictions = automl_model.predict(
                X_exogen=self.test_multivar[[self.ts_column] + self.preds], # Not needed for ARIMA
                forecast_period=self.forecast_period,
                model="ARIMA",
                simple=False
            )
            # print(test_predictions)
            # self.assertIsNone(
            #     np.testing.assert_array_equal(
            #         test_predictions.columns.values, self.expected_pred_col_names
            #     )
            # )
            
        if automl_model.get_model_build('SARIMAX') is not None:
            # Simple forecast with forecast window = one used in training
            # Using named model
            test_predictions = automl_model.predict(
                forecast_period=self.forecast_period,
                X_exogen=self.test_multivar[[self.ts_column] + self.preds],
                model="SARIMAX"                
            )

            assert_series_equal(test_predictions.round(6), self.forecast_gold_sarimax_multivar_external_test)
            
            # Simple forecast with forecast window != one used in training
            # Using named model
            test_predictions = automl_model.predict(
                forecast_period=6,
                X_exogen=self.test_multivar.iloc[0:6][[self.ts_column] + self.preds],
                model="SARIMAX"                
            )
            assert_series_equal(test_predictions.round(6), self.forecast_gold_sarimax_multivar_external_test_10)

            # Complex forecasts (returns confidence intervals, etc.)
            test_predictions = automl_model.predict(
                forecast_period=self.forecast_period,
                X_exogen=self.test_multivar[[self.ts_column] + self.preds],
                model="SARIMAX",
                simple=False                
            )
            self.assertIsNone(
                np.testing.assert_array_equal(
                    test_predictions.columns.values, self.expected_pred_col_names
                )
            )

            # Checking missing exogenous variables
            with self.assertRaises(ValueError):
                test_predictions = automl_model.predict(
                    forecast_period=self.forecast_period,
                    model="SARIMAX"                
                )

            # Checking missing columns from exogenous variables
            with self.assertRaises(ValueError):
                test_multivar_temp = self.test_multivar.copy(deep=True)
                test_multivar_temp.rename(columns={'Marketing Expense': 'Incorrect Column'}, inplace=True)
                test_predictions = automl_model.predict(
                        forecast_period=self.forecast_period,
                        X_exogen=test_multivar_temp,
                        model="SARIMAX"                
                    )

            test_predictions = automl_model.predict(
                forecast_period=10,
                X_exogen=self.test_multivar[[self.ts_column] + self.preds],
                model="SARIMAX"                
            )
            assert_series_equal(test_predictions.round(6), self.forecast_gold_sarimax_multivar_external_test)


            
        if automl_model.get_model_build('VAR') is not None:
            # Simple forecast with forecast window = one used in training
            # Using named model
            test_predictions = automl_model.predict(
                X_exogen=self.test_multivar[[self.ts_column] + self.preds], # Not needed for VAR
                forecast_period=self.forecast_period,
                model="VAR"
            )
            assert_series_equal(test_predictions.round(6), self.forecast_gold_var_multivar_series)
            
            # Simple forecast with forecast window != one used in training
            # Using named model
            test_predictions = automl_model.predict(
                X_exogen=self.test_multivar[[self.ts_column] + self.preds], # Not needed for VAR
                forecast_period=10,
                model="VAR"
            )
            assert_series_equal(test_predictions.round(6), self.forecast_gold_var_multivar_series_10)

            # Complex forecasts (returns confidence intervals, etc.)
            test_predictions = automl_model.predict(
                X_exogen=self.test_multivar[[self.ts_column] + self.preds], # Not needed for VAR
                forecast_period=self.forecast_period,
                model="VAR",
                simple=False
            )
            self.assertIsNone(
                np.testing.assert_array_equal(
                    test_predictions.columns.values, self.expected_pred_col_names
                )
            )

        if automl_model.get_model_build('ML') is not None:
            # Simple forecast with forecast window = one used in training
            # Using named model
            test_predictions = automl_model.predict(
                forecast_period=self.forecast_period,
                X_exogen=self.test_multivar[[self.ts_column] + self.preds],
                model="ML"                
            )
            assert_series_equal(test_predictions.round(6), self.forecast_gold_ml_multivar_external_test)
            
            # Simple forecast with forecast window != one used in training
            # Using named model
            test_predictions = automl_model.predict(
                forecast_period=6,
                X_exogen=self.test_multivar.iloc[0:6][[self.ts_column] + self.preds],
                model="ML"                
            )
            assert_series_equal(test_predictions.round(6), self.forecast_gold_ml_multivar_external_test_10)

            # Complex forecasts (returns confidence intervals, etc.)
            test_predictions = automl_model.predict(
                forecast_period=self.forecast_period,
                X_exogen=self.test_multivar[[self.ts_column] + self.preds],
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
                np.round(ml_dict.get('Prophet').get('forecast')[0]['yhat'],6),
                self.forecast_gold_prophet_multivar_internal_val_cv_fold2
            ),
            "(Multivar Test) Prophet Forecast does not match up with expected values."
        )

        self.assertEqual(
            round(ml_dict.get('Prophet').get('rmse')[0],8), self.rmse_gold_prophet_multivar_cv_fold2,
            "(Multivar Test) Prophet RMSE does not match up with expected values.")

        ################################
        #### Checking ARIMA Results ####
        ################################
        
        # https://stackoverflow.com/questions/19387608/attributeerror-rint-when-using-numpy-round
        self.assertIsNone(
            np.testing.assert_array_equal(
                np.round(ml_dict.get('ARIMA').get('forecast')['mean'].values.astype(np.double), 8),
                self.forecast_gold_arima_uni_multivar
            ),
            "(Multivar Test) ARIMA Forecast does not match up with expected values."
        )

        self.assertEqual(
            round(ml_dict.get('ARIMA').get('rmse'),8), self.rmse_gold_arima_uni_multivar,
            "(Multivar Test) ARIMA RMSE does not match up with expected values.")

        ##################################
        #### Checking SARIMAX Results ####
        ##################################
        self.assertIsNone(
            np.testing.assert_array_equal(
                np.round(ml_dict.get('SARIMAX').get('forecast')[0]['mean'].values.astype(np.double), 6),
                self.forecast_gold_sarimax_multivar_internal_val
            ),
            "(Multivar Test) SARIMAX Forecast does not match up with expected values."
        )
        
        self.assertEqual(
            round(ml_dict.get('SARIMAX').get('rmse')[0], 6), self.rmse_gold_sarimax_multivar,
            "(Multivar Test) SARIMAX RMSE does not match up with expected values.")
               
        ##############################
        #### Checking VAR Results ####
        ##############################
        self.assertIsNone(
            np.testing.assert_array_equal(
                np.round(ml_dict.get('VAR').get('forecast')['mean'].values.astype(np.double), 8),
                self.forecast_gold_var_multivar
            ),
            "(Multivar Test) VAR Forecast does not match up with expected values."
        )
        
        self.assertEqual(
            round(ml_dict.get('VAR').get('rmse'),8), self.rmse_gold_var_multivar,
            "(Multivar Test) VAR RMSE does not match up with expected values.")

        #############################
        #### Checking ML Results ####
        #############################

        self.assertListEqual(
            ml_dict.get('ML').get('forecast'), self.forecast_gold_ml_multivar_internal_val,
            "(Multivar Test) ML Forecast does not match up with expected values.")
        
        self.assertEqual(
            round(ml_dict.get('ML').get('rmse')[0], 6), self.rmse_gold_ml_multivar,
            "(Multivar Test) ML RMSE does not match up with expected values.")

    # @unittest.skip
    def test_auto_ts_univar_ns_SARIMAX(self):
        """
        test to check functionality of the auto_ts function (univariate models with non seasonal SARIMAX)
        """

        print("\n\n" + "*"*50)
        print("Performing Unit Test: 'test_auto_ts_univar_ns_SARIMAX'")
        print("*"*50 + "\n\n")

        import numpy as np  # type: ignore
        from auto_ts import AutoTimeSeries as ATS
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
        test_predictions = automl_model.predict(forecast_period=self.forecast_period)  
        
        ml_dict = automl_model.get_ml_dict()
        
        leaderboard_gold = pd.DataFrame(
            {
                'name': ['Prophet', 'ARIMA', 'SARIMAX', 'PyFlux', 'VAR', 'ML'],
                'rmse':[
                    self.rmse_gold_prophet_univar_cv_fold2,
                    self.rmse_gold_arima_uni_multivar,
                    self.rmse_gold_sarimax_univar,
                    math.inf,
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
        
        self.assertTrue(
            isinstance(automl_model.get_model('SARIMAX'), SARIMAXResultsWrapper),
            "SARIMAX model does not match the expected type."
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
            assert_series_equal(test_predictions.round(6), self.forecast_gold_prophet_univar_external_test_cv)        

            # Simple forecast with forecast window != one used in training
            # Using default (best model)
            test_predictions = automl_model.predict(forecast_period=10)
            assert_series_equal(test_predictions.round(6), self.forecast_gold_prophet_univar_external_test_10_cv)   

            # Simple forecast with forecast window = one used in training
            # Using named model
            test_predictions = automl_model.predict(
                forecast_period=self.forecast_period,
                model="Prophet"
            )
            assert_series_equal(test_predictions.round(6), self.forecast_gold_prophet_univar_external_test_cv)        

            # Simple forecast with forecast window != one used in training
            # Using named model
            test_predictions = automl_model.predict(
                forecast_period=10,
                model="Prophet")
            assert_series_equal(test_predictions.round(6), self.forecast_gold_prophet_univar_external_test_10_cv)        

            # Complex forecasts (returns confidence intervals, etc.)
            test_predictions = automl_model.predict(
                forecast_period=self.forecast_period,
                model="Prophet",
                simple=False
            )
            # print(test_predictions)
            # self.assertIsNone(
            #     np.testing.assert_array_equal(
            #         test_predictions.columns.values, self.expected_pred_col_names_prophet
            #     )
            # )
        

        if automl_model.get_model_build('ARIMA') is not None:
            # Simple forecast with forecast window = one used in training
            # Using named model
            test_predictions = automl_model.predict(
                forecast_period=self.forecast_period,
                model="ARIMA"
            )
            assert_series_equal(test_predictions.round(6), self.forecast_gold_arima_uni_multivar_series) 
            
            # Simple forecast with forecast window != one used in training
            # Using named model
            test_predictions = automl_model.predict(
                forecast_period=10,
                model="ARIMA"
            )
            assert_series_equal(test_predictions.round(6), self.forecast_gold_arima_uni_multivar_series_10) 

            # Complex forecasts (returns confidence intervals, etc.)
            test_predictions = automl_model.predict(
                forecast_period=self.forecast_period,
                model="ARIMA",
                simple=False
            )
            # print(test_predictions)
            # self.assertIsNone(
            #     np.testing.assert_array_equal(
            #         test_predictions.columns.values, self.expected_pred_col_names
            #     )
            # )
            
        if automl_model.get_model_build('SARIMAX') is not None:
            # Simple forecast with forecast window = one used in training
            # Using named model
            test_predictions = automl_model.predict(
                forecast_period=self.forecast_period,
                model="SARIMAX"
            )
            assert_series_equal(test_predictions.round(6), self.forecast_gold_sarimax_univar_external_test)
            
            # Simple forecast with forecast window != one used in training
            # Using named model
            test_predictions = automl_model.predict(
                forecast_period=10,
                model="SARIMAX"
            )
            assert_series_equal(test_predictions.round(6), self.forecast_gold_sarimax_univar_external_test_10)

            # Complex forecasts (returns confidence intervals, etc.)
            test_predictions = automl_model.predict(
                forecast_period=self.forecast_period,
                model="SARIMAX",
                simple=False
            )
            self.assertIsNone(
                np.testing.assert_array_equal(
                    test_predictions.columns.values, self.expected_pred_col_names
                )
            )
            
        if automl_model.get_model_build('VAR') is not None:
            # Simple forecast with forecast window = one used in training
            # Using named model
            test_predictions = automl_model.predict(
                forecast_period=self.forecast_period,
                model="VAR"
            )
            assert_series_equal(test_predictions.round(6), self.forecast_gold_var_univar_series)
            
            # Simple forecast with forecast window != one used in training
            # Using named model
            test_predictions = automl_model.predict(
                forecast_period=10,
                model="VAR"
            )
            assert_series_equal(test_predictions.round(6), self.forecast_gold_var_univar_series_10)

            # Complex forecasts (returns confidence intervals, etc.)
            test_predictions = automl_model.predict(
                forecast_period=self.forecast_period,
                model="VAR",
                simple=False
            )
            self.assertIsNone(
                np.testing.assert_array_equal(
                    test_predictions.columns.values, self.expected_pred_col_names
                )
            )

        if automl_model.get_model_build('ML') is not None:
            # Simple forecast with forecast window = one used in training
            # Using named model
            test_predictions = automl_model.predict(
                forecast_period=self.forecast_period,
                model="ML"                
            )
            assert_series_equal(test_predictions.round(6), self.forecast_gold_ml_univar_series)
            
            # Simple forecast with forecast window != one used in training
            # Using named model
            test_predictions = automl_model.predict(
                forecast_period=6,
                model="ML"                
            )
            assert_series_equal(test_predictions.round(6), self.forecast_gold_ml_univar_series_10)

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
        self.assertIsNone(automl_model.get_model_build('VAR'), "Expected Univar VAR model to be None but did not get None.")
        self.assertIsNone(automl_model.get_model_build('ML'), "Expected Univar ML model to be None but did not get None.")


        ##################################
        #### Checking Prophet Results ####
        ##################################
        self.assertIsNone(
            np.testing.assert_array_equal(
                np.round(ml_dict.get('Prophet').get('forecast')[0]['yhat'],6),
                self.forecast_gold_prophet_univar_internal_val_cv_fold2
            ),
            "(Univar Test) Prophet Forecast does not match up with expected values."
        )

        self.assertEqual(
            round(ml_dict.get('Prophet').get('rmse')[0],8), self.rmse_gold_prophet_univar_cv_fold2,
            "(Univar Test) Prophet RMSE does not match up with expected values.")

        ################################
        #### Checking ARIMA Results ####
        ################################
        
        # https://stackoverflow.com/questions/19387608/attributeerror-rint-when-using-numpy-round
        self.assertIsNone(
            np.testing.assert_array_equal(
                np.round(ml_dict.get('ARIMA').get('forecast')['mean'].values.astype(np.double), 8),
                self.forecast_gold_arima_uni_multivar
            ),
            "(Univar Test) ARIMA Forecast does not match up with expected values."
        )

        self.assertEqual(
            round(ml_dict.get('ARIMA').get('rmse'),8), self.rmse_gold_arima_uni_multivar,
            "(Univar Test) ARIMA RMSE does not match up with expected values.")

        ##################################
        #### Checking SARIMAX Results ####
        ##################################
        self.assertIsNone(
            np.testing.assert_array_equal(
                np.round(ml_dict.get('SARIMAX').get('forecast')[0]['mean'].values.astype(np.double), 8),
                self.forecast_gold_sarimax_univar_internal_val
            ),
            "(Univar Test) SARIMAX Forecast does not match up with expected values."
        )
        
        self.assertEqual(
            round(ml_dict.get('SARIMAX').get('rmse')[0],8), self.rmse_gold_sarimax_univar,
            "(Univar Test) SARIMAX RMSE does not match up with expected values.")
               
        ##############################
        #### Checking VAR Results ####
        ##############################
        self.assertEqual(
            ml_dict.get('VAR').get('forecast'), self.forecast_gold_var_univar,
            "(Univar Test) VAR Forecast does not match up with expected values."
        )
        
        self.assertEqual(
            round(ml_dict.get('VAR').get('rmse'),8), self.rmse_gold_var_univar,
            "(Univar Test) VAR RMSE does not match up with expected values.")

        #############################
        #### Checking ML Results ####
        #############################
        
        self.assertEqual(
            ml_dict.get('ML').get('forecast'), self.forecast_gold_ml_univar,
            "(Univar Test) ML Forecast does not match up with expected values."
        )
        
        self.assertEqual(
            round(ml_dict.get('ML').get('rmse'),8), self.rmse_gold_ml_univar,
            "(Univar Test) ML RMSE does not match up with expected values."
        )
    
    # @unittest.skip
    def test_auto_ts_multivar_seasonal_SARIMAX(self):
        """
        test to check functionality of the auto_ts function (multivariate with seasonal SARIMAX)
        """

        print("\n\n" + "*"*50)
        print("Performing Unit Test: 'test_auto_ts_multivar_seasonal_SARIMAX'")
        print("*"*50 + "\n\n")

        import numpy as np  # type: ignore
        from auto_ts import AutoTimeSeries as ATS
        # TODO: seasonal_period argument does not make a difference. Commenting out for now.
        automl_model = ATS(
            score_type='rmse', forecast_period=self.forecast_period, time_interval='Month',
            non_seasonal_pdq=None, seasonality=True, seasonal_period=12,
            # non_seasonal_pdq=None, seasonality=True, seasonal_period=3,
            model_type='SARIMAX',
            verbose=0)
        
        automl_model.fit(
            traindata=self.train_multivar,
            ts_column=self.ts_column,
            target=self.target,
            cv=None #,
            # sep=self.sep
            )
        
        test_predictions = automl_model.predict(
            forecast_period=self.forecast_period,
            X_exogen=self.test_multivar[[self.ts_column] + self.preds],
        )  
        
        ml_dict = automl_model.get_ml_dict()
        
        leaderboard_gold = pd.DataFrame(
            {
                'name':['SARIMAX'],
                'rmse':[self.rmse_gold_sarimax_multivar_s12]
            }
        )
        assert_frame_equal(automl_model.get_leaderboard().reset_index(drop=True).round(6), leaderboard_gold)

        if automl_model.get_model_build('SARIMAX') is not None:
            # Simple forecast with forecast window = one used in training
            # Using named model
            test_predictions = automl_model.predict(
                forecast_period=self.forecast_period,
                X_exogen=self.test_multivar[[self.ts_column] + self.preds],
                model="SARIMAX"                
            )

            assert_series_equal(test_predictions.round(6), self.forecast_gold_sarimax_multivar_external_test_s12)
            
            # Simple forecast with forecast window != one used in training
            # Using named model
            test_predictions = automl_model.predict(
                forecast_period=6,
                X_exogen=self.test_multivar.iloc[0:6][[self.ts_column] + self.preds],
                model="SARIMAX"                
            )
            assert_series_equal(test_predictions.round(6), self.forecast_gold_sarimax_multivar_external_test_10_s12)

            # Complex forecasts (returns confidence intervals, etc.)
            test_predictions = automl_model.predict(
                forecast_period=self.forecast_period,
                X_exogen=self.test_multivar[[self.ts_column] + self.preds],
                model="SARIMAX",
                simple=False                
            )
            self.assertIsNone(
                np.testing.assert_array_equal(
                    test_predictions.columns.values, self.expected_pred_col_names
                )
            )

            # Checking missing exogenous variables
            with self.assertRaises(ValueError):
                test_predictions = automl_model.predict(
                    forecast_period=self.forecast_period,
                    # X_exogen=self.test_multivar[self.preds],
                    model="SARIMAX"                
                )

            # Checking missing columns from exogenous variables
            with self.assertRaises(ValueError):
                test_multivar_temp = self.test_multivar.copy(deep=True)
                test_multivar_temp.rename(columns={'Marketing Expense': 'Incorrect Column'}, inplace=True)
                test_predictions = automl_model.predict(
                        forecast_period=self.forecast_period,
                        X_exogen=test_multivar_temp,
                        model="SARIMAX"                
                    )

            test_predictions = automl_model.predict(
                forecast_period=10,
                X_exogen=self.test_multivar[[self.ts_column] + self.preds],
                model="SARIMAX"                
            )
            assert_series_equal(test_predictions.round(6), self.forecast_gold_sarimax_multivar_external_test_s12)


        ##################################
        #### Checking SARIMAX Results ####
        ##################################
        self.assertIsNone(
            np.testing.assert_array_equal(
                np.round(ml_dict.get('SARIMAX').get('forecast')[0]['mean'].values.astype(np.double), 6),
                self.forecast_gold_sarimax_multivar_internal_val_s12
            ),
            "(Multivar Test) SARIMAX Forecast does not match up with expected values."
        )
        
        self.assertEqual(
            round(ml_dict.get('SARIMAX').get('rmse')[0], 6), self.rmse_gold_sarimax_multivar_s12,
            "(Multivar Test) SARIMAX RMSE does not match up with expected values.")

    # @unittest.skip
    def test_auto_ts_multivar_seasonal_SARIMAX_withCV(self):
        """
        test to check functionality of the auto_ts function (multivariate with seasonal SARIMAX)
        """
        print("\n\n" + "*"*50)
        print("Performing Unit Test: 'test_auto_ts_multivar_seasonal_SARIMAX_withCV'")
        print("*"*50 + "\n\n")

        import numpy as np  # type: ignore
        from auto_ts import AutoTimeSeries as ATS

        # Reduced seasonality from 12 to 3 since we are doing CV. There is not enough data
        # to use seasonality of 12 along with CV since the first fold only has 24 train obs.
        automl_model = ATS(
            score_type='rmse', forecast_period=self.forecast_period, time_interval='Month',
            non_seasonal_pdq=None, seasonality=True, seasonal_period=3,
            model_type='SARIMAX',
            verbose=0)
        
        automl_model.fit(
            traindata=self.train_multivar,
            ts_column=self.ts_column,
            target=self.target,
            cv=2 #,
            # sep=self.sep
            )
        
        test_predictions = automl_model.predict(
            forecast_period=self.forecast_period,
            X_exogen=self.test_multivar[[self.ts_column] + self.preds],
        )  
        
        ml_dict = automl_model.get_ml_dict()
        
        leaderboard_gold = pd.DataFrame(
            {
                'name':['SARIMAX'],
                'rmse':[self.rmse_gold_sarimax_multivar_s12_cv]
            }
        )
        assert_frame_equal(automl_model.get_leaderboard().reset_index(drop=True).round(6), leaderboard_gold)

        if automl_model.get_model_build('SARIMAX') is not None:
            # Simple forecast with forecast window = one used in training
            # Using named model
            test_predictions = automl_model.predict(
                forecast_period=self.forecast_period,
                X_exogen=self.test_multivar[[self.ts_column] + self.preds],
                model="SARIMAX"                
            )

            assert_series_equal(test_predictions.round(6), self.forecast_gold_sarimax_multivar_external_test_s3_cv)
            
            # Simple forecast with forecast window != one used in training
            # Using named model
            test_predictions = automl_model.predict(
                forecast_period=6,
                X_exogen=self.test_multivar.iloc[0:6][[self.ts_column] + self.preds],
                model="SARIMAX"                
            )
            assert_series_equal(test_predictions.round(6), self.forecast_gold_sarimax_multivar_external_test_10_s3_cv)

            # Complex forecasts (returns confidence intervals, etc.)
            test_predictions = automl_model.predict(
                forecast_period=self.forecast_period,
                X_exogen=self.test_multivar[[self.ts_column] + self.preds],
                model="SARIMAX",
                simple=False                
            )
            self.assertIsNone(
                np.testing.assert_array_equal(
                    test_predictions.columns.values, self.expected_pred_col_names
                )
            )

            # Checking missing exogenous variables
            with self.assertRaises(ValueError):
                test_predictions = automl_model.predict(
                    forecast_period=self.forecast_period,
                    # X_exogen=self.test_multivar[self.preds],
                    model="SARIMAX"                
                )

            # Checking missing columns from exogenous variables
            with self.assertRaises(ValueError):
                test_multivar_temp = self.test_multivar.copy(deep=True)
                test_multivar_temp.rename(columns={'Marketing Expense': 'Incorrect Column'}, inplace=True)
                test_predictions = automl_model.predict(
                        forecast_period=self.forecast_period,
                        X_exogen=test_multivar_temp,
                        model="SARIMAX"                
                    )

            test_predictions = automl_model.predict(
                forecast_period=10,
                X_exogen=self.test_multivar[[self.ts_column] + self.preds],
                model="SARIMAX"                
            )
            assert_series_equal(test_predictions.round(6), self.forecast_gold_sarimax_multivar_external_test_s3_cv)


        ##################################
        #### Checking SARIMAX Results ####
        ##################################
        self.assertIsNone(
            np.testing.assert_array_equal(
                np.round(ml_dict.get('SARIMAX').get('forecast')[0]['mean'].values.astype(np.double), 6),
                self.forecast_gold_sarimax_multivar_internal_val_s3_cv_fold1
            ),
            "(Multivar Test) SARIMAX Forecast does not match up with expected values --> Fold 1."
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                np.round(ml_dict.get('SARIMAX').get('forecast')[1]['mean'].values.astype(np.double), 6),
                self.forecast_gold_sarimax_multivar_internal_val_s3_cv_fold2
            ),
            "(Multivar Test) SARIMAX Forecast does not match up with expected values --> Fold 2."
        )
        
        self.assertEqual(
            round(ml_dict.get('SARIMAX').get('rmse')[0], 6), self.rmse_gold_sarimax_multivar_s3_cv_fold1,
            "(Multivar Test) SARIMAX RMSE does not match up with expected values --> Fold 1.")
        self.assertEqual(
            round(ml_dict.get('SARIMAX').get('rmse')[1], 6), self.rmse_gold_sarimax_multivar_s3_cv_fold2,
            "(Multivar Test) SARIMAX RMSE does not match up with expected values --> Fold 2.")
               
    # @unittest.skip
    def test_subset_of_models(self):
        """
        test to check functionality of the training with only a subset of models
        """

        print("\n\n" + "*"*50)
        print("Performing Unit Test: 'test_subset_of_models'")
        print("*"*50 + "\n\n")

        import numpy as np  # type: ignore
        from auto_ts import AutoTimeSeries as ATS

        automl_model = ATS(
            score_type='rmse', forecast_period=self.forecast_period, time_interval='Month',
            non_seasonal_pdq=None, seasonality=False, seasonal_period=12,
            model_type=['SARIMAX', 'ML'],
            verbose=0)
        automl_model.fit(
            traindata=self.train_multivar,
            ts_column=self.ts_column,
            target=self.target,
            cv=None,
            sep=self.sep)
        print(automl_model.get_leaderboard())
        leaderboard_gold = pd.DataFrame(
            {
                'name': ['ML', 'SARIMAX'],
                'rmse': [self.rmse_gold_ml_multivar, self.rmse_gold_sarimax_multivar] 
            }
        )
        assert_frame_equal(automl_model.get_leaderboard().reset_index(drop=True).round(6), leaderboard_gold)

        automl_model = ATS(
            score_type='rmse', forecast_period=self.forecast_period, time_interval='Month',
            non_seasonal_pdq=None, seasonality=False, seasonal_period=12,
            model_type=['SARIMAX', 'bogus', 'ML'],
            verbose=0)
        automl_model.fit(
            traindata=self.train_multivar,
            ts_column=self.ts_column,
            target=self.target,
            cv=None,
            sep=self.sep)
        print(automl_model.get_leaderboard())
        
        leaderboard_gold = pd.DataFrame(
            {
                'name': ['ML', 'SARIMAX'],
                'rmse': [self.rmse_gold_ml_multivar, self.rmse_gold_sarimax_multivar] 
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

    # @unittest.skip
    def test_passing_list_instead_of_str(self):
        """
        Tests passing models as a list insteasd of a string
        """
        
        print("\n\n" + "*"*50)
        print("Performing Unit Test: 'test_passing_list_instead_of_str'")
        print("*"*50 + "\n\n")

        import numpy as np  # type: ignore
        from auto_ts import AutoTimeSeries as ATS

        automl_model = ATS(
            score_type='rmse', forecast_period=self.forecast_period, time_interval='Month',
            non_seasonal_pdq=None, seasonality=False, seasonal_period=12,
            model_type=['SARIMAX', 'ML'],
            verbose=0)
        automl_model.fit(
            traindata=self.train_multivar,
            ts_column=[self.ts_column],
            target=[self.target],
            cv=None,
            sep=self.sep)
        print(automl_model.get_leaderboard())
        leaderboard_models = np.array(['ML', 'SARIMAX'])

        np.testing.assert_array_equal(automl_model.get_leaderboard()['name'].values, leaderboard_models)

    # @unittest.skip
    def test_cv_retreival_plotting(self):
        """
        Tests CV Scores retreival and plotting
        """

        print("\n\n" + "*"*50)
        print("Performing Unit Test: 'test_cv_retreival_plotting'")
        print("*"*50 + "\n\n")

        import numpy as np  # type: ignore
        from auto_ts import AutoTimeSeries as ATS


        automl_model = ATS(
            score_type='rmse', forecast_period=self.forecast_period, time_interval='Month',
            non_seasonal_pdq=None, seasonality=False, seasonal_period=12,
            model_type=['ML', 'SARIMAX'],
            verbose=0)
        automl_model.fit(
            traindata=self.train_multivar,
            ts_column=self.ts_column,
            target=self.target,
            cv=2,
            sep=self.sep)        

        cv_scores_gold = pd.DataFrame(
            {
                'Model': ['SARIMAX', 'SARIMAX', 'ML', 'ML'],
                # 'CV Scores': [73.2824, 185.705, 93.0305, 67.304]
                'CV Scores': [73.2824, 185.705, 107.2037, 81.7406]  # With more engineered features (AutoViML)
            }
        )
        cv_scores = automl_model.get_cv_scores()
        assert_frame_equal(cv_scores, cv_scores_gold)   
        
        automl_model.plot_cv_scores()

    # @unittest.skip
    def test_prophet_multivar_standalone_noCV(self):
        """
        test to check functionality Prophet with CV
        """
        print("\n\n" + "*"*50)
        print("Performing Unit Test: 'test_prophet_multivar_standalone_noCV'")
        print("*"*50 + "\n\n")

        import numpy as np  # type: ignore
        from auto_ts import AutoTimeSeries as ATS

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
                np.round(ml_dict.get('Prophet').get('forecast')[0]['yhat'],6),
                self.forecast_gold_prophet_multivar_internal_val_cv_fold2
            )
        )

        # Validation RMSE
        self.assertEqual(
            round(ml_dict.get('Prophet').get('rmse')[0],8), self.rmse_gold_prophet_multivar_cv_fold2
        )
        
        ##############################
        #### Extarnal Test Checks ####
        ##############################

        # Simple forecast with forecast window = one used in training
        # Using named model
        test_predictions = automl_model.predict(
            X_exogen=self.test_multivar[[self.ts_column] + self.preds], 
            forecast_period=self.forecast_period,
            model="Prophet"
        )

        print(f"FB Prophet Predictions: {test_predictions}")

        assert_series_equal(test_predictions.round(6), self.forecast_gold_prophet_multivar_external_test_cv)        

        # Simple forecast with forecast window != one used in training
        # Using named model
        test_predictions = automl_model.predict(
            X_exogen=self.test_multivar[[self.ts_column] + self.preds], 
            forecast_period=10,
            model="Prophet")
        assert_series_equal(test_predictions.round(6), self.forecast_gold_prophet_multivar_external_test_10_cv)        

    # @unittest.skip
    def test_prophet_multivar_standalone_withCV(self):
        """
        test to check functionality Prophet with CV
        """
        print("\n\n" + "*"*50)
        print("Performing Unit Test: 'test_prophet_multivar_standalone_withCV'")
        print("*"*50 + "\n\n")

        import numpy as np  # type: ignore
        from auto_ts import AutoTimeSeries as ATS

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
                np.round(ml_dict.get('Prophet').get('forecast')[0]['yhat'],6),
                self.forecast_gold_prophet_multivar_internal_val_cv_fold1
            )
        )
        self.assertIsNone(
            np.testing.assert_array_equal(
                np.round(ml_dict.get('Prophet').get('forecast')[1]['yhat'],6),
                self.forecast_gold_prophet_multivar_internal_val_cv_fold2
            )
        )

        # Validation RMSE
        self.assertEqual(
            round(ml_dict.get('Prophet').get('rmse')[0],8), self.rmse_gold_prophet_multivar_cv_fold1
        )
        self.assertEqual(
            round(ml_dict.get('Prophet').get('rmse')[1],8), self.rmse_gold_prophet_multivar_cv_fold2
        )
        
        ##############################
        #### Extarnal Test Checks ####
        ##############################

        # Simple forecast with forecast window = one used in training
        # Using named model
        test_predictions = automl_model.predict(
            X_exogen=self.test_multivar[[self.ts_column] + self.preds], 
            forecast_period=self.forecast_period,
            model="Prophet"
        )
        assert_series_equal(test_predictions.round(6), self.forecast_gold_prophet_multivar_external_test_cv)        

        # Simple forecast with forecast window != one used in training
        # Using named model
        test_predictions = automl_model.predict(
            X_exogen=self.test_multivar[[self.ts_column] + self.preds], 
            forecast_period=10,
            model="Prophet")
        assert_series_equal(test_predictions.round(6), self.forecast_gold_prophet_multivar_external_test_10_cv)        
    
    def test_ml_standalone_noCV(self):
        """
        Testing ML Standalone without CV
        """

        print("\n\n" + "*"*50)
        print("Performing Unit Test: 'test_ml_standalone_noCV'")
        print("*"*50 + "\n\n")

        import numpy as np  # type: ignore
        from auto_ts import AutoTimeSeries as ATS

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
        print(automl_model.get_leaderboard())

    def test_ml_standalone_withCV(self):
        """
        Testing ML Standalone with CV
        """
        print("\n\n" + "*"*50)
        print("Performing Unit Test: 'test_ml_standalone_withCV'")
        print("*"*50 + "\n\n")

        import numpy as np  # type: ignore
        from auto_ts import AutoTimeSeries as ATS

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
                X_exogen=self.test_multivar[[self.ts_column] + self.preds],
                model="ML"                
            )
            assert_series_equal(test_predictions.round(6), self.forecast_gold_ml_multivar_external_test_cv)
            
            # Simple forecast with forecast window != one used in training
            # Using named model
            test_predictions = automl_model.predict(
                forecast_period=6,
                X_exogen=self.test_multivar.iloc[0:6][[self.ts_column] + self.preds],
                model="ML"                
            )
            assert_series_equal(test_predictions.round(6), self.forecast_gold_ml_multivar_external_test_10_cv)

            # Complex forecasts (returns confidence intervals, etc.)
            test_predictions = automl_model.predict(
                forecast_period=self.forecast_period,
                X_exogen=self.test_multivar[[self.ts_column] + self.preds],
                model="ML",
                simple=False                
            )
            self.assertIsNone(
                np.testing.assert_array_equal(
                    test_predictions.columns.values, self.expected_pred_col_names
                )
            )

            # For ML forecast window is ignored (horizon only depends on the length of X_exogen)
            # Hence even though we specify forecast_period of 10, it only takes 8 here
            test_predictions = automl_model.predict(
                forecast_period=10,
                X_exogen=self.test_multivar[[self.ts_column] + self.preds],
                model="ML"                
            )
            assert_series_equal(test_predictions.round(6), self.forecast_gold_ml_multivar_external_test_cv)

        # RMSE check for each fold
        self.assertEqual(
            round(ml_dict.get('ML').get('rmse')[0], 8), self.rmse_gold_ml_multivar_cv_fold1,
            "(Multivar Test) ML RMSE does not match up with expected values --> Fold 1.")
        self.assertEqual(
            round(ml_dict.get('ML').get('rmse')[1], 8), self.rmse_gold_ml_multivar_cv_fold2,
            "(Multivar Test) ML RMSE does not match up with expected values --> Fold 2.")

        
     
if __name__ == '__main__':
    unittest.main()