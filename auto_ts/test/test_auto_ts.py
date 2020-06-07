import unittest

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

        self.train = dft[:40]
        self.test = dft[40:]


    def test_auto_ts(self):
        """
        test to check functionality of the auto_ts function
        """
        import numpy as np  # type: ignore
        import auto_ts as AT
        ml_dict = AT.Auto_Timeseries(
            self.train, self.ts_column,
            self.target, self.sep,  score_type='rmse', forecast_period=8,
            time_interval='Month', non_seasonal_pdq=None, seasonality=False,
            seasonal_period=12, seasonal_PDQ=None, model_type='best',
            verbose=0
        )

        # https://stackoverflow.com/questions/25348532/can-python-pickle-lambda-functions
        # import dill  # the code below will fail without this line
        # import pickle
        # with open('ml_dict.pickle', 'wb') as handle:
        #     pickle.dump(ml_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # with open('ml_dict.pickle', 'rb') as handle:
        #     ml_dict_gold = pickle.load(handle)

        # self.assertDictEqual(ml_dict, ml_dict_gold, "The generated ml_dict does not match the golden dictionary.")

        print(ml_dict)

        ##################################
        #### Checking Prophet Results ####
        ##################################
        forecast_gold = np.array([
            397.43339084, 394.26439651, 475.13957452, 552.65076563, 606.16644019, 593.80751381, 660.50017734, 660.71231806,
            507.50617922, 428.91362082, 394.42162318, 460.58145002, 414.11761317, 411.79136617, 513.90686713, 548.44630982,
            625.04519821, 601.93200453, 692.72711895, 713.80546701, 509.75238742, 452.27192698, 417.23842764, 489.43692325,
            464.33630331, 463.7618856 , 554.96050385, 607.84174268, 680.80447392, 665.27454447, 751.95122103, 769.70733192,
            583.80971329, 520.80174673, 487.2960147 , 558.92329098, 527.98407913, 528.04537126, 615.77231537, 682.98205328,
            749.06124155, 751.07726213, 796.89236612, 783.20673348,689.69812976, 595.71342586, 569.48660003, 635.88437079
            ])
        
        self.assertIsNone(
            np.testing.assert_array_equal(np.round(ml_dict.get('FB_Prophet').get('forecast'),8), forecast_gold),
            "Prophet Forecast does not match up with expected values."
        )

        rmse_gold = 27.01794672
        self.assertEqual(round(ml_dict.get('FB_Prophet').get('rmse'),8), rmse_gold, "Prophet RMSE does not match up with expected values.")

        ################################
        #### Checking ARIMA Results ####
        ################################
        forecast_gold = np.array([
            801.78660584, 743.16044526, 694.38764549, 684.72931967,
            686.70229610, 692.13402266, 698.59426282, 705.36034762            
            ])
        
        # https://stackoverflow.com/questions/19387608/attributeerror-rint-when-using-numpy-round
        self.assertIsNone(
            np.testing.assert_array_equal(np.round(ml_dict.get('ARIMA').get('forecast')['mean'].values.astype(np.double), 8), forecast_gold),
            "ARIMA Forecast does not match up with expected values."
        )

        rmse_gold = 169.00016628
        self.assertEqual(round(ml_dict.get('ARIMA').get('rmse'),8), rmse_gold, "ARIMA RMSE does not match up with expected values.")

        ##################################
        #### Checking SARIMAX Results ####
        ##################################
        forecast_gold = np.array([
            803.31673726, 762.46093997, 718.3581931,  711.42130506,
            719.36254603, 732.70981867, 747.57645435, 762.47349398
            ])
        
        self.assertIsNone(
            np.testing.assert_array_equal(np.round(ml_dict.get('SARIMAX').get('forecast')['mean'].values.astype(np.double), 8), forecast_gold),
            "SARIMAX Forecast does not match up with expected values."
        )
        

        rmse_gold = 193.49650578
        self.assertEqual(round(ml_dict.get('SARIMAX').get('rmse'),8), rmse_gold, "SARIMAX RMSE does not match up with expected values.")
               
        ##############################
        #### Checking VAR Results ####
        ##############################
        forecast_gold = np.array([
            741.37790864, 676.23341949, 615.53872102, 571.7977285,
            546.95278336, 537.34223069, 537.4744872,  542.30739271
            ])

        self.assertIsNone(
            np.testing.assert_array_equal(np.round(ml_dict.get('VAR').get('forecast')['mean'].values.astype(np.double), 8), forecast_gold),
            "VAR Forecast does not match up with expected values."
        )
        
        rmse_gold = 112.4770318
        self.assertEqual(round(ml_dict.get('VAR').get('rmse'),8), rmse_gold, "VAR RMSE does not match up with expected values.")

        #############################
        #### Checking ML Results ####
        #############################
        forecast_gold = np.array([
            475.24, 455.72, 446.58, 450.82,
            453.76, 457.96, 475.04, 564.78
            ])

        self.assertIsNone(
            np.testing.assert_array_equal(np.round(ml_dict.get('ML').get('forecast').astype(np.double), 2), forecast_gold),
            "ML Forecast does not match up with expected values."
        )
        
        rmse_gold = 94.94981174
        self.assertEqual(round(ml_dict.get('ML').get('rmse'),8), rmse_gold, "VAR RMSE does not match up with expected values.")
       


     
if __name__ == '__main__':
    unittest.main()