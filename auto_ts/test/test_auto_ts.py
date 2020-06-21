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

        self.train_multivar = dft[:40]
        self.test_multivar = dft[40:]

        self.train_univar = dft[:40][[self.ts_column, self.target]]
        self.test_univar = dft[40:][[self.ts_column, self.target]]

        self.forecast_period = 8

        # self.expected_pred_col_names_prophet = np.array(['mean', 'mean_se', 'mean_ci_lower', 'mean_ci_upper'])
        self.expected_pred_col_names_stats = np.array(['mean', 'mean_se', 'mean_ci_lower', 'mean_ci_upper'])
        
        ################################
        #### Prophet Golden Results ####
        ################################

        #### UNIVARIATE ####

        self.forecast_gold_prophet_univar = np.array([
            397.43339084, 394.26439651, 475.13957452, 552.65076563, 606.16644019, 593.80751381, 660.50017734, 660.71231806,
            507.50617922, 428.91362082, 394.42162318, 460.58145002, 414.11761317, 411.79136617, 513.90686713, 548.44630982,
            625.04519821, 601.93200453, 692.72711895, 713.80546701, 509.75238742, 452.27192698, 417.23842764, 489.43692325,
            464.33630331, 463.7618856 , 554.96050385, 607.84174268, 680.80447392, 665.27454447, 751.95122103, 769.70733192,
            583.80971329, 520.80174673, 487.2960147 , 558.92329098, 527.98407913, 528.04537126, 615.77231537, 682.98205328,
            749.06124155, 751.07726213, 796.89236612, 783.20673348,689.69812976, 595.71342586, 569.48660003, 635.88437079
            ])

        results = [
            749.061242, 751.077262, 796.892366, 783.206733,
            689.698130, 595.713426, 569.486600, 635.884371           
            ]
        index = np.arange(40, 48)

        self.forecast_gold_prophet_univar_series = pd.Series(
                data = results,
                index = index
            )
        self.forecast_gold_prophet_univar_series.name = 'yhat'

        results = results + [576.473786, 581.275889]
        index = np.arange(40, 50)
            
        self.forecast_gold_prophet_univar_series_10 = pd.Series(
                data = results, 
                index = index
            )
        self.forecast_gold_prophet_univar_series_10.name = 'yhat'

        self.rmse_gold_prophet_univar = 27.01794672


        #### MULTIVARIATE ####
        # TODO: Change multivariate model results after adding capability for multivariate models
        self.forecast_gold_prophet_multivar = np.array([
            397.43339084, 394.26439651, 475.13957452, 552.65076563, 606.16644019, 593.80751381, 660.50017734, 660.71231806,
            507.50617922, 428.91362082, 394.42162318, 460.58145002, 414.11761317, 411.79136617, 513.90686713, 548.44630982,
            625.04519821, 601.93200453, 692.72711895, 713.80546701, 509.75238742, 452.27192698, 417.23842764, 489.43692325,
            464.33630331, 463.7618856 , 554.96050385, 607.84174268, 680.80447392, 665.27454447, 751.95122103, 769.70733192,
            583.80971329, 520.80174673, 487.2960147 , 558.92329098, 527.98407913, 528.04537126, 615.77231537, 682.98205328,
            749.06124155, 751.07726213, 796.89236612, 783.20673348,689.69812976, 595.71342586, 569.48660003, 635.88437079
            ])

        results = [
            749.061242, 751.077262, 796.892366, 783.206733,
            689.698130, 595.713426, 569.486600, 635.884371           
            ]
        index = np.arange(40, 48)

        self.forecast_gold_prophet_multivar_series = pd.Series(
                data = results,
                index = index
            )
        self.forecast_gold_prophet_multivar_series.name = 'yhat'

        results = results + [576.473786, 581.275889]
        index = np.arange(40, 50)
            
        self.forecast_gold_prophet_multivar_series_10 = pd.Series(
                data = results, 
                index = index
            )
        self.forecast_gold_prophet_multivar_series_10.name = 'yhat'

        self.rmse_gold_prophet_multivar = 27.01794672


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
        
        ################################
        #### SARIMAX Golden Results ####
        ################################

        #### UNIVARIATE ####

        results = [
            803.31673726, 762.46093997, 718.3581931,  711.42130506,
            719.36254603, 732.70981867, 747.57645435, 762.47349398            
            ]

        index = pd.to_datetime([
            '2013-09-01', '2013-10-01', '2013-11-01', '2013-12-01',
            '2014-01-01', '2014-02-01', '2014-03-01', '2014-04-01'
            ])
        
        self.forecast_gold_sarimax_univar = np.array(results)
        
        self.forecast_gold_sarimax_univar_series = pd.Series(
                data = results,
                index = index
            )
        self.forecast_gold_sarimax_univar_series.name = 'mean'

        results = results + [776.914078, 790.809653]
        index = pd.to_datetime([
            '2013-09-01', '2013-10-01', '2013-11-01', '2013-12-01',
            '2014-01-01', '2014-02-01', '2014-03-01', '2014-04-01',
            '2014-05-01', '2014-06-01'
            ])
            
        self.forecast_gold_sarimax_univar_series_10 = pd.Series(
                data = results,
                index = index
            )
        self.forecast_gold_sarimax_univar_series_10.name = 'mean'

        self.rmse_gold_sarimax_univar = 193.49650578

        #### MULTIVARIATE ####
        # TODO: Change multivariate model results after adding capability for multivariate models
        
        results = [
            803.31673726, 762.46093997, 718.3581931,  711.42130506,
            719.36254603, 732.70981867, 747.57645435, 762.47349398            
            ]
        index = pd.to_datetime([
            '2013-09-01', '2013-10-01', '2013-11-01', '2013-12-01',
            '2014-01-01', '2014-02-01', '2014-03-01', '2014-04-01'
            ])
        
        self.forecast_gold_sarimax_multivar = np.array(results)
        
        self.forecast_gold_sarimax_multivar_series = pd.Series(
                data = results,
                index = index
            )
        self.forecast_gold_sarimax_multivar_series.name = 'mean'

        results = results + [776.914078, 790.809653]
        index = pd.to_datetime([
            '2013-09-01', '2013-10-01', '2013-11-01', '2013-12-01',
            '2014-01-01', '2014-02-01', '2014-03-01', '2014-04-01',
            '2014-05-01', '2014-06-01'
            ])
            
        self.forecast_gold_sarimax_multivar_series_10 = pd.Series(
                data = results,
                index = index
            )
        self.forecast_gold_sarimax_multivar_series_10.name = 'mean'

        self.rmse_gold_sarimax_multivar = 193.49650578

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

        self.forecast_gold_ml_univar = None
        self.rmse_gold_ml_univar = math.inf

        # This was with previous ML predict method where we had leakage
        # self.forecast_gold_ml_multivar = np.array([
        #     475.24, 455.72, 446.58, 450.82,
        #     453.76, 457.96, 475.04, 564.78
        #     ])

        # This is with new ML predict method without leakage
        self.forecast_gold_ml_multivar = np.array([
            475.24, 444.8 , 437.7 , 446.44,
            449.88, 476.68, 622.04, 640.48
            ])

        # self.rmse_gold_ml_multivar = 94.94981174 # This was with previous ML predict method where we had leakage
        self.rmse_gold_ml_multivar = 76.03743322 # This is with new ML predict method without leakage
        



    #@unittest.skip
    def test_auto_ts_multivar(self):
        """
        test to check functionality of the auto_ts function
        """
        import numpy as np  # type: ignore
        from auto_ts.auto_ts import AutoTimeseries as ATS
        automl_model = ATS(
            score_type='rmse', forecast_period=self.forecast_period, time_interval='Month',
            non_seasonal_pdq=None, seasonality=False, seasonal_period=12, seasonal_PDQ=None,
            model_type='best',
            verbose=0)
        automl_model.fit(self.train_multivar, self.ts_column, self.target, self.sep)
        test_predictions = automl_model.predict(forecast_period=self.forecast_period)  # TODO: pass the test dataframe (exogen)
        print("Test Predictions from outside AutoML:")
        print(test_predictions)
        ml_dict = automl_model.get_ml_dict()
        print("Final Dictionary...")
        print(ml_dict)

        print(automl_model.get_leaderboard())
        leaderboard_gold = pd.DataFrame(
            {
                'name':['FB_Prophet', 'ML', 'VAR', 'ARIMA', 'SARIMAX', 'PyFlux'],
                # 'rmse':[27.017947, 94.949812, 112.477032, 169.000166, 193.496506, math.inf]  # This was with previous ML predict method where we had leakage
                'rmse':[27.017947, 76.037433, 112.477032, 169.000166, 193.496506, math.inf] # This is with new ML predict method without leakage
            }
        )
        assert_frame_equal(automl_model.get_leaderboard().reset_index(drop=True).round(6), leaderboard_gold)

        self.assertEqual(
            automl_model.get_best_model_name(), "FB_Prophet",
            "Best model name does not match expected value."
        )
        self.assertTrue(
            isinstance(automl_model.get_best_model(), Prophet), 
            "Best model does not match expected value."
        )
        # print(f"Best Model: {automl_model.get_best_model()}")

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
            # print("-"*50)
            # print("Predictions with Best Model (Prophet)")
            # print("-"*50)

            # Simple forecast with forecast window = one used in training
            # Using default (best model)
            test_predictions = automl_model.predict(forecast_period=self.forecast_period)
            assert_series_equal(test_predictions.round(6), self.forecast_gold_prophet_multivar_series)        

            # Simple forecast with forecast window != one used in training
            # Using default (best model)
            test_predictions = automl_model.predict(forecast_period=10)
            assert_series_equal(test_predictions.round(6), self.forecast_gold_prophet_multivar_series_10)   

            # Simple forecast with forecast window = one used in training
            # Using named model
            test_predictions = automl_model.predict(
                forecast_period=self.forecast_period,
                model="FB_Prophet"
            )
            assert_series_equal(test_predictions.round(6), self.forecast_gold_prophet_multivar_series)        

            # Simple forecast with forecast window != one used in training
            # Using named model
            test_predictions = automl_model.predict(
                forecast_period=10,
                model="FB_Prophet")
            assert_series_equal(test_predictions.round(6), self.forecast_gold_prophet_multivar_series_10)        

            # Complex forecasts (returns confidence intervals, etc.)
            test_predictions = automl_model.predict(
                forecast_period=self.forecast_period,
                model="FB_Prophet",
                simple=False
            )
            # print(test_predictions)
            # self.assertIsNone(
            #     np.testing.assert_array_equal(
            #         test_predictions.columns.values, self.expected_pred_col_names_prophet
            #     )
            # )
        

        if automl_model.get_model_build('ARIMA') is not None:
            # print("-"*50)
            # print("Predictions with ARIMA Model")
            # print("-"*50)
            
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
            #         test_predictions.columns.values, self.expected_pred_col_names_stats
            #     )
            # )
            
        if automl_model.get_model_build('SARIMAX') is not None:
            # print("-"*50)
            # print("Predictions with SARIMAX Model")
            # print("-"*50)

            # Simple forecast with forecast window = one used in training
            # Using named model
            test_predictions = automl_model.predict(
                forecast_period=self.forecast_period,
                model="SARIMAX"
            )
            assert_series_equal(test_predictions.round(6), self.forecast_gold_sarimax_multivar_series)
            
            # Simple forecast with forecast window != one used in training
            # Using named model
            test_predictions = automl_model.predict(
                forecast_period=10,
                model="SARIMAX"
            )
            assert_series_equal(test_predictions.round(6), self.forecast_gold_sarimax_multivar_series_10)

            # Complex forecasts (returns confidence intervals, etc.)
            test_predictions = automl_model.predict(
                forecast_period=self.forecast_period,
                model="SARIMAX",
                simple=False
            )
            self.assertIsNone(
                np.testing.assert_array_equal(
                    test_predictions.columns.values, self.expected_pred_col_names_stats
                )
            )
            
        if automl_model.get_model_build('VAR') is not None:
            # print("-"*50)
            # print("Predictions with VAR Model")
            # print("-"*50)

            # Simple forecast with forecast window = one used in training
            # Using named model
            test_predictions = automl_model.predict(
                forecast_period=self.forecast_period,
                model="VAR"
            )
            assert_series_equal(test_predictions.round(6), self.forecast_gold_var_multivar_series)
            
            # Simple forecast with forecast window != one used in training
            # Using named model
            test_predictions = automl_model.predict(
                forecast_period=10,
                model="VAR"
            )
            assert_series_equal(test_predictions.round(6), self.forecast_gold_var_multivar_series_10)

            # Complex forecasts (returns confidence intervals, etc.)
            test_predictions = automl_model.predict(
                forecast_period=self.forecast_period,
                model="VAR",
                simple=False
            )
            self.assertIsNone(
                np.testing.assert_array_equal(
                    test_predictions.columns.values, self.expected_pred_col_names_stats
                )
            )
        
        ##################################
        #### Checking Prophet Results ####
        ##################################
        self.assertIsNone(
            np.testing.assert_array_equal(
                np.round(ml_dict.get('FB_Prophet').get('forecast'),8),
                self.forecast_gold_prophet_multivar
            ),
            "(Multivar Test) Prophet Forecast does not match up with expected values."
        )

        self.assertEqual(
            round(ml_dict.get('FB_Prophet').get('rmse'),8), self.rmse_gold_prophet_multivar,
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
                np.round(ml_dict.get('SARIMAX').get('forecast')['mean'].values.astype(np.double), 8),
                self.forecast_gold_sarimax_multivar
            ),
            "(Multivar Test) SARIMAX Forecast does not match up with expected values."
        )
        
        self.assertEqual(
            round(ml_dict.get('SARIMAX').get('rmse'),8), self.rmse_gold_sarimax_multivar,
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
        
        self.assertIsNone(
            np.testing.assert_array_equal(
                np.round(ml_dict.get('ML').get('forecast').astype(np.double), 2),
                self.forecast_gold_ml_multivar
            ),
            "(Multivar Test) ML Forecast does not match up with expected values."
        )
        
        self.assertEqual(
            round(ml_dict.get('ML').get('rmse'),8), self.rmse_gold_ml_multivar,
            "(Multivar Test) ML RMSE does not match up with expected values.")


    #@unittest.skip
    def test_auto_ts_univar(self):
        """
        test to check functionality of the auto_ts function (univariate models)
        """
        import numpy as np  # type: ignore
        from auto_ts.auto_ts import AutoTimeseries as ATS
        automl_model = ATS(
            score_type='rmse', forecast_period=self.forecast_period, time_interval='Month',
            non_seasonal_pdq=None, seasonality=False, seasonal_period=12, seasonal_PDQ=None,
            model_type='best',
            verbose=0)
        automl_model.fit(self.train_univar, self.ts_column, self.target, self.sep)
        test_predictions = automl_model.predict(forecast_period=self.forecast_period)  
        print("Test Predictions from outside AutoML:")
        print(test_predictions)
        ml_dict = automl_model.get_ml_dict()
        print("Final Dictionary...")
        print(ml_dict)

        print(automl_model.get_leaderboard())
        leaderboard_gold = pd.DataFrame(
            {
                'name':['FB_Prophet', 'ARIMA', 'SARIMAX', 'PyFlux', 'VAR', 'ML'],
                'rmse':[27.017947, 169.000166, 193.496506, math.inf, math.inf, math.inf] 
            }
        )
        assert_frame_equal(automl_model.get_leaderboard().reset_index(drop=True).round(6), leaderboard_gold)

        self.assertEqual(
            automl_model.get_best_model_name(), "FB_Prophet",
            "Best model name does not match expected value."
        )
        self.assertTrue(
            isinstance(automl_model.get_best_model(), Prophet), 
            "Best model does not match expected value."
        )
        # print(f"Best Model: {automl_model.get_best_model()}")

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


        # if automl_model.get_best_model_build() is not None:
        #     print("-"*50)
        #     print("Predictions with Best Model (Prophet)")
        #     print("-"*50)
        #     print(f"Type: {type(automl_model.get_best_model_build().predict())}")
        #     print(automl_model.get_best_model_build().predict())
        #     print(automl_model.get_best_model_build().predict(forecast_period=10))

        # if automl_model.get_model_build('ARIMA') is not None:
        #     print("-"*50)
        #     print("Predictions with ARIMA Model")
        #     print("-"*50)
        #     print(f"Type: {type(automl_model.get_model_build('ARIMA').predict())}")
        #     print(automl_model.get_model_build('ARIMA').predict())
        #     print(automl_model.get_model_build('ARIMA').predict(forecast_period=10))

        # if automl_model.get_model_build('SARIMAX') is not None:
        #     print("-"*50)
        #     print("Predictions with SARIMAX Model")
        #     print("-"*50)
        #     print(f"Type: {type(automl_model.get_model_build('SARIMAX').predict())}")
        #     print(automl_model.get_model_build('SARIMAX').predict())
        #     print(automl_model.get_model_build('SARIMAX').predict(forecast_period=10))

        # if automl_model.get_model_build('VAR') is not None:
        #     print("-"*50)
        #     print("Predictions with VAR Model")
        #     print("-"*50)
        #     print(f"Type: {type(automl_model.get_model_build('VAR').predict())}")
        #     print(automl_model.get_model_build('VAR').predict())
        #     print(automl_model.get_model_build('VAR').predict(forecast_period=10))

        if automl_model.get_best_model_build() is not None:
            # print("-"*50)
            # print("Predictions with Best Model (Prophet)")
            # print("-"*50)

            # Simple forecast with forecast window = one used in training
            # Using default (best model)
            test_predictions = automl_model.predict(forecast_period=self.forecast_period)
            assert_series_equal(test_predictions.round(6), self.forecast_gold_prophet_univar_series)        

            # Simple forecast with forecast window != one used in training
            # Using default (best model)
            test_predictions = automl_model.predict(forecast_period=10)
            assert_series_equal(test_predictions.round(6), self.forecast_gold_prophet_univar_series_10)   

            # Simple forecast with forecast window = one used in training
            # Using named model
            test_predictions = automl_model.predict(
                forecast_period=self.forecast_period,
                model="FB_Prophet"
            )
            assert_series_equal(test_predictions.round(6), self.forecast_gold_prophet_univar_series)        

            # Simple forecast with forecast window != one used in training
            # Using named model
            test_predictions = automl_model.predict(
                forecast_period=10,
                model="FB_Prophet")
            assert_series_equal(test_predictions.round(6), self.forecast_gold_prophet_univar_series_10)        

            # Complex forecasts (returns confidence intervals, etc.)
            test_predictions = automl_model.predict(
                forecast_period=self.forecast_period,
                model="FB_Prophet",
                simple=False
            )
            # print(test_predictions)
            # self.assertIsNone(
            #     np.testing.assert_array_equal(
            #         test_predictions.columns.values, self.expected_pred_col_names_prophet
            #     )
            # )
        

        if automl_model.get_model_build('ARIMA') is not None:
            # print("-"*50)
            # print("Predictions with ARIMA Model")
            # print("-"*50)
            
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
            #         test_predictions.columns.values, self.expected_pred_col_names_stats
            #     )
            # )
            
        if automl_model.get_model_build('SARIMAX') is not None:
            # print("-"*50)
            # print("Predictions with SARIMAX Model")
            # print("-"*50)

            # Simple forecast with forecast window = one used in training
            # Using named model
            test_predictions = automl_model.predict(
                forecast_period=self.forecast_period,
                model="SARIMAX"
            )
            assert_series_equal(test_predictions.round(6), self.forecast_gold_sarimax_univar_series)
            
            # Simple forecast with forecast window != one used in training
            # Using named model
            test_predictions = automl_model.predict(
                forecast_period=10,
                model="SARIMAX"
            )
            assert_series_equal(test_predictions.round(6), self.forecast_gold_sarimax_univar_series_10)

            # Complex forecasts (returns confidence intervals, etc.)
            test_predictions = automl_model.predict(
                forecast_period=self.forecast_period,
                model="SARIMAX",
                simple=False
            )
            self.assertIsNone(
                np.testing.assert_array_equal(
                    test_predictions.columns.values, self.expected_pred_col_names_stats
                )
            )
            
        if automl_model.get_model_build('VAR') is not None:
            # print("-"*50)
            # print("Predictions with VAR Model")
            # print("-"*50)

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
                    test_predictions.columns.values, self.expected_pred_col_names_stats
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
                np.round(ml_dict.get('FB_Prophet').get('forecast'),8),
                self.forecast_gold_prophet_univar
            ),
            "(Univar Test) Prophet Forecast does not match up with expected values."
        )

        self.assertEqual(
            round(ml_dict.get('FB_Prophet').get('rmse'),8), self.rmse_gold_prophet_univar,
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
                np.round(ml_dict.get('SARIMAX').get('forecast')['mean'].values.astype(np.double), 8),
                self.forecast_gold_sarimax_univar
            ),
            "(Univar Test) SARIMAX Forecast does not match up with expected values."
        )
        
        self.assertEqual(
            round(ml_dict.get('SARIMAX').get('rmse'),8), self.rmse_gold_sarimax_univar,
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
       

    


     
if __name__ == '__main__':
    unittest.main()