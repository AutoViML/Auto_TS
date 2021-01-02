<!DOCTYPE html>
<html>
<head>
</head>
<body>

![auto-ts](logo.png)

<h1 id="auto-ts">Auto_TS: Auto_TimeSeries</h1>
<p style="font-family:verdana">Automatically build multiple Time Series models using a Single Line of Code.</p>
<p>Auto_timeseries is a complex model building utility for time series data. Since it automates many
Tasks involved in a complex endeavor, it assumes many intelligent defaults. But you can change them.
Auto_Timeseries will rapidly build predictive models based on Statsmodels ARIMA, Seasonal ARIMA
and Scikit-Learn ML. It will automatically select the best model which gives best score specified.
</p>
<p>New version 0.0.25 onwards changes the syntax of Auto_TimeSeries to be more like scikit-learn (fit and predict syntax). You will have to initialize an object and then call fit with your data and then predict again with data. Hope this makes it easier to remember and use.</p>
<h2 id="introduction">Introduction</h2>
<p>Auto_TimeSeries enables you to build and select multiple time series models using techniques such as ARIMA, SARIMAX, VAR, decomposable (trend+seasonality+holidays) models, and ensemble machine learning models.</p>
<p>Auto_TimeSeries is an Automated ML library for time series data. Auto_TimeSeries was initially conceived and developed by <a href="https://www.linkedin.com/in/ram-seshadri-nyc-nj/">Ram Seshadri</a> and was significantly expanded in functionality and scope and upgraded to its present status by <a href="https://github.com/ngupta23">Nikhil Gupta</a>.</p>
<p>auto-ts.Auto_TimeSeries() is the main function that you will call with your train data. You can then choose what kind of models you want: stats, ml or FB prophet based model. You can also tell it to automatically select the best model based on the scoring parameter you want it to be based on. It will return the best model and a dictionary containing predictions for the number of forecast_periods you mentioned (default=2).</p>
<h2 id="installation-instructions">INSTALLATION INSTRUCTIONS</h2>
<ol>
<li>Use “pip install auto-ts”</li>
<li>Use “pip3 install auto-ts” if the above doesn’t work</li>
<li>pip install git+git://github.com/AutoViML/Auto_TS</li>
</ol>
<h2 id="run">RUN</h2>
<p> First you need to import auto_timeseries from auto_ts library:<br>
<code>
from auto_ts import auto_timeseries
</code>
</p>
<p>
Second, Initialize an auto_timeseries model object which will hold all your parameters:</p>
<p><code>
model = auto_timeseries(
            score_type='rmse',
            time_interval='Month',
            non_seasonal_pdq=None, seasonality=False,
            seasonal_period=12,
            model_type=['Prophet'],
            verbose=2)
</code></p>
Here are how the input parameters defined:<br>
<ol>
<li><b>score_type (default='rmse')</b>: The metric used for scoring the models. Type is string.
Currently only the following two types are supported:
(1) "rmse": Root Mean Squared Error (RMSE)
(2) "normalized_rmse": Ratio of RMSE to the standard deviation of actuals</li>
<li><b>time_interval (default is None)</b>: Used to indicate the frequency at which the data is collected.
This is used for two purposes (1) in building the Prophet model and (2) used to impute the seasonal period for SARIMAX in case it is not provided by the user (None). Type is String.
We use the following <a href="https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases">pandas date range frequency</a> aliases that Prophet uses to make the prediction dataframe. <p>Hence, please note that these are the list of allowed aliases for frequency:
                      ['B','C','D','W','M','SM','BM','CBM',
                     'MS','SMS','BMS','CBMS','Q','BQ','QS','BQS',
                     'A,Y','BA,BY','AS,YS','BAS,BYS','BH',
                     'H','T,min','S','L,ms','U,us','N']

For a start, you can test the following codes for your data and see how the results are:
<ul>
<li>'MS', 'M', 'SM', 'BM', 'CBM', 'SMS', 'BMS' for monthly frequency data</li>
<li> 'D', 'B', 'C' for daily frequency data </li>
<li> 'W' for weekly frequency data </li>
<li> 'Q', 'BQ', 'QS', 'BQS' for quarterly frequency data </li>
<li> 'A,Y', 'BA,BY', 'AS,YS', 'BAS,YAS' for yearly frequency data </li>
<li> 'BH', 'H', 'h' for hourly frequency data </li>
<li> 'T,min' for minute frequency data </li>
<li> 'S', 'L,milliseconds', 'U,microseconds', 'N,nanoseconds' for second frequency data </li>
</ul>
Or you can leave it as None and auto_timeseries will try and impute it.
</li>
<li><b>non_seasonal_pdq (default = (3,1,3))</b>: Indicates the maximum value of (p, d, q) to be used in the search for statistical ARIMA models.
If None, then the following values are assumed max_p = 3, max_d = 1, max_q = 3. Type is Tuple.</li>
<li><b>seasonality (default=False)</b>: Used in the building of the SARIMAX model only at this time. True or False. Type is bool.</li>
<li><b>seasonal_period (default is None)</b>: Indicates the seasonal period in your data. This depends on the peak (or valley) period that occurs regularly in your data.
Used in the building of the SARIMAX model only at this time.
There is no impact of this argument if seasonality is set to False
If None, the program will try to infer this from the time_interval (frequency) of the data
We assume the following as defaults but feel free to change them.
(1) If frequency is Monthly, then seasonal_period is assumed to be 12
(1) If frequency is Daily, then seasonal_period is assumed to be 30 (but it could be 7)
(1) If frequency is Weekly, then seasonal_period is assumed to be 52
(1) If frequency is Quarterly, then seasonal_period is assumed to be 4
(1) If frequency is Yearly, then seasonal_period is assumed to be 1
(1) If frequency is Hourly, then seasonal_period is assumed to be 24
(1) If frequency is Minutes, then seasonal_period is assumed to be 60
(1) If frequency is Seconds, then seasonal_period is assumed to be 60
Type is integer</li>
<li><b>conf_int (default=0.95)</b>: Confidence Interval for building the Prophet model. Default: 0.95. Type is float.</li>
<li><b>model_type (default: 'stats'</b>: The type(s) of model to build. Default to building only statistical models
Can be a string or a list of models. Allowed values are:
'best', 'prophet', 'stats', 'ARIMA', 'SARIMAX', 'VAR', 'ML'.
"prophet" will build a model using FB Prophet -> this means you must have FB Prophet installed
"stats" will build statsmodels based ARIMA, SARIMAX and VAR models
"ML" will build a machine learning model using Random Forests provided explanatory vars are given
'best' will try to build all models and pick the best one
If a list is provided, then only those models will be built
WARNING: "best" might take some time for large data sets. We recommend that you
choose a small sample from your data set before attempting to run entire data.
Type can be either: [string, list]
</li>
<li><b>verbose (default=0)</b>: Indicates the verbosity of printing (Default = 0). Type is integer.</li>
</ol>
The next step after defining the model object is to fit it with some real data:
<p>
<code>
model.fit(
            traindata=train_data,
            ts_column=ts_column,
            target=target,
            cv=5,
            sep=","
        )
</code></p>
<br>Here are how the parameters defined:
<ul>
<li><b>traindata (required)</b>: It can be either a dataframe or a file. You must give the name of the file along with its data path in case if a file. It also accepts a pandas dataframe in case you already have a dataframe loaded in your notebook.</li>
<li><b>ts_column (required)</b>: name of the datetime column in your dataset (it could be a name of column or index number in the columns index)</li>
<li><b>target (required)</b>: name of the column you   are trying to predict. Target could also be the only column in your dataset</li>
<li><b>cv (default=5)</b>: You can enter any integer for the number of folds you want in your cross validation data set.
</li>
<li><b>sep (default=",")</b>: Sep is the separator in your traindata file. If your separator is ",", "\t", ";", make sure you enter it here. If not, it is ignored.</li>
</ul>
The next step after training the model object is to make some predictions with test data:<br>
<p>
<code>
predictions = model.predict(
            testdata = can be either a dataframe or an integer signifying the forecast_period,
            model = 'best' or any other string that stands for the trained model
        )  
</code></p>
Here are how the parameters are defined. You can choose to send either testdata in the form of a dataframe or send in an integer to decide how many periods you want to forecast.  You need only
<ul>
<li><b>testdata (required)</b>: It can be either a dataframe containing test data or you can use an integer standing for the forecast_period (you want).</li>
<li><b>model (optional, default = 'best')</b>: The name of the model you want to use among the many different models you have trained. Remember that the default is the best model. But you can choose any model that you want to forecast with. Type is String.</li>
</ul>
<h2 id="requirements">Requirements</h2>
<p>tscv</p>
<p>scikit-learn</p>
<p>FB Prophet</p>
<p>statsmodels</p>
<p>pmdarima</p>
<h1 id="license">License:</h1>
<p>Apache License 2.0</p>
<h2 id="recommendations">Recommendations</h2>
<ul>
<li>We recommend that you choose a small sample from your data set before attempting to run entire data. and the evaluation metric so it can select the best model. Currently models within “stats” are compared using AIC and BIC. However, models across different types are compared using RMSE. The results of models are shown using RMSE and Normalized RMSE (ratio of RMSE to the standard deviation of actuals).</li>
<li>You must clean the data and not have any missing values. Make sure the target variable is numeric, otherwise, it won’t run. If there is more than one target variable in your data set, just specify only one for now, and if you know the time interval that is in your data, you can specify it. Otherwise it auto-ts will try to infer the time interval on its own.</li>
<li>If you give Auto_Timeseries a different time interval than what the data has, it will automatically resample the data to the given time interval and use the mean of the target for the resampled period.</li>
<li>Notice that except for filename and ts_column input arguments, which are required, all other arguments are optional.</li>
<li>Note that optionally you can give a separator for the data in your file. Default is comma (",").</li>
<li>“time_interval” options are any codes that you can find in this page below.
<a href="https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases">Pandas date-range frequency aliases</a>
</li>
<li>Optionally, you can give seasonal_period as any integer that measures the seasonality in the data. If not given, seasonal_period is assumed automatically as follows: Months = 12, Days = 30, Weeks = 52, Qtr = 4, Year = 1, Hours = 24, Minutes = 60 and Seconds = 60.</li>
<li>If you want to give your own non-seasonal order, please input it as non_seasonal_pdq and for seasonal order, use seasonal_PDQ as the input. Use tuples. For example, seasonal_PDQ = (2,1,2) and non_seasonal_pdq = (0,0,3). It will accept only tuples. The default is None and Auto_Timeseries will automatically search for the best p,d,q (for Non Seasonal) and P, D, Q (for Seasonal) orders by searching for all parameters from 0 to 12 for each value of p,d,q and 0-3 for each P, Q and 0-1 for D.</li>
</ul>
<h2 id="disclaimer">DISCLAIMER:</h2>
<p>This is not an Officially supported Google project.</p>
<h2 id="copyright">Copyright</h2>
<p>© Google</p>
</body>
</html>
