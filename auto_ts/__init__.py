##########################################################
#Defining AUTO_TIMESERIES here
##########################################################
module_type = 'Running' if  __name__ == "__main__" else 'Imported'
version_number = '0.0.22'
# TODO: Fix based on new interface
print("""Running Auto Timeseries version: %s...Call by using:
        auto_ts.Auto_Timeseries(traindata, ts_column,
                            target, sep,  score_type='rmse', forecast_period=5,
                            time_interval='Month', non_seasonal_pdq=None, seasonality=False,
                            seasonal_period=12, seasonal_PDQ=None, model_type='stats',
                            verbose=1)
    To run three models from Stats, ML and FB Prophet, set model_type='best'""" % version_number)
print("To remove previous versions, perform 'pip uninstall auto_ts'")
print('To get the latest version, perform "pip install auto_ts --no-cache-dir --ignore-installed"')


