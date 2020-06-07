<h1 id="auto-ts">Auto-TS</h1>
<p>Automatically build multiple Time Series models using a Single Line of Code.</p>
<h2 id="introduction">Introduction</h2>
<p>auto-ts is an Automated ML library for time series data.</p>
<p>auto-ts enables you to build and select multiple time series models using techniques such as ARIMA, SARIMAX, VAR, decomposable (trend+seasonality+holidays) models, and ensemble machine learning models.</p>
<p>auto-ts,Auto_TimeSeries() is the main function that you will call with your train data. You can then choose what kind of models you want: stats, ml or FB prophet based model. You can also tell it to automatically select the best model based on the scoring parameter you want it to be based on. It will return the best model and a dictionary containing predictions for the number of forecast_periods you mentioned (default=2).</p>
<h2 id="installation-instructions">INSTALLATION INSTRUCTIONS</h2>
<ol>
<li>Use “pip install auto-ts”</li>
<li>Use “pip3 install auto-ts” if the above doesn’t work</li>
<li>pip install git+git://github.com/AutoViML/Auto_TS</li>
</ol>
<h2 id="run">RUN</h2>
<p>auto_ts.Auto_Timeseries(traindata, ts_column,<br>
target, sep,  score_type=‘rmse’, forecast_period=5,<br>
time_interval=‘Month’, non_seasonal_pdq=None, seasonality=False,<br>
seasonal_period=12, seasonal_PDQ=None, model_type=‘stats’,<br>
verbose=1)</p>
<h2 id="requirements">Requirements</h2>
<p>PyFlux</p>
<p>FB Prophet</p>
<p>statsmodels</p>
<h1 id="license">License:</h1>
<p>Apache License 2.0</p>
<h1 id="inputs">INPUTS:</h1>
<ul>
<li>trainfile: name of the file along with its data path or a dataframe.<br>
It accepts either a pandas dataframe or name of the file and its data path.</li>
<li>ts_column: name of the datetime column in your<br>
dataset (it could be name or number)</li>
<li>target: name of the column you   are trying to predict. Target could also be the only column in your<br>
data</li>
<li>score_type: ‘rmse’ is the default. You can choose among “mae”,<br>
“mse” and “rmse”.   forecast_period: default is 2. How many periods out do you want to forecast? It should be an integer</li>
<li>time_interval: default is “Month”. What is the time period in your data set. Options are: “day”,   ‘month’,‘week’,‘year’,‘quarter’ etc.</li>
<li>model_type:   default is “stats”. Choice is between “stats”, “prophet”, “ml”, and “best”.
<ul>
<li>“stats” will build statsmodels based ARIMA&lt; SARIMAX and VAR models</li>
<li>“ml” will build a machine learning model using Random Forests provided explanatory vars are given</li>
<li>“prophet” will build a model using FB Prophet -&gt; this means you must have FB Prophet installed</li>
<li>“best” will build three of the best models from above which might take some time for large data sets.</li>
</ul>
</li>
</ul>
<h2 id="recommendations">Recommendations</h2>
<ul>
<li>We recommend that you choose a small sample from your data set before attempting to run entire data. and the evaluation metric so it can select the best model. Currently models within “stats” are compared using AIC and BIC. However, models across different types are compared using RMSE. The results of models are shown using RMSE and Normalized RMSE (ratio of RMSE to the standard deviation of actuals).</li>
<li>You must clean the data and not have any missing values. Make sure the target variable is numeric, otherwise, it won’t run. If there is more than one target variable in your data set, just specify only one for now, and if you know the time interval that is in your data, you can specify it. Otherwise it auto-ts will try to infer the time interval on its own.</li>
<li>If you give Auto_Timeseries a different time interval than what the data has, it will automatically resample the data to the given time interval and use the mean of the target for the resampled period.</li>
<li>Notice that except for filename and ts_column input arguments, which are required, all other arguments are optional.</li>
<li>Note that optionally you can give a separator for the data in your file. Default is comma (",").</li>
<li>“time_interval” options are: ‘Days’, ‘Weeks’, ‘Months’, ‘Qtr’, ‘Year’, ‘Minutes’, ‘Hours’, ‘Seconds’.</li>
<li>Optionally, you can give seasonal_period as any integer that measures the seasonality in the data. If not given, seasonal_period is assumed automatically as follows: Months = 12, Days = 30, Weeks = 52, Qtr = 4, Year = 1, Hours = 24, Minutes = 60 and Seconds = 60.</li>
<li>If you want to give your own non-seasonal order, please input it as non_seasonal_pdq and for seasonal order, use seasonal_PDQ as the input. Use tuples. For example, seasonal_PDQ = (2,1,2) and non_seasonal_pdq = (0,0,3). It will accept only tuples. The default is None and Auto_Timeseries will automatically search for the best p,d,q (for Non Seasonal) and P, D, Q (for Seasonal) orders by searching for all parameters from 0 to 12 for each value of p,d,q and 0-3 for each P, Q and 0-1 for D.</li>
</ul>
<h2 id="disclaimer">DISCLAIMER:</h2>
<p>This is not an Officially supported Google project.</p>
<h2 id="copyright">Copyright</h2>
<p>© Google</p>
