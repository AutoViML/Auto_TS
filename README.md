<h1 id="auto-ts">Auto_TS: Auto_TimeSeries</h1>
<p style="font-family:verdana">Automatically build multiple Time Series models using a Single Line of Code. Now updated with Dask.

![auto-ts](images/logo.png)

`auto_timeseries` is a complex model building utility for time series data. Since it automates many
Tasks involved in a complex endeavor, it assumes many intelligent defaults. But you can change them.
Auto_Timeseries will rapidly build predictive models based on Statsmodels ARIMA, Seasonal ARIMA, Prophet
and Scikit-Learn ML. It will automatically select the best model which gives best score specified.

# Table of Contents
<ul>
<li><a href="#Latest">Latest updates</a></li>
<li><a href="#Citation">Citation</a></li>
<li><a href="#introduction">What is auto_ts</a></li>
<li><a href="#install">How to install auto_ts</a></li>
<li><a href="#usage">Usage</a></li>
<li><a href="#requirements">Requirements</a></li>
<li><a href="#tips">Tips</a></li>
<li><a href="#license">License</a></li>
<li><a href="#copyright">Copyright</a></li>
<li><a href="#disclaimer">Disclaimer</a></li>
</ul>

## Latest
If you are looking for the latest and greatest updates about our library, check out our [updates page](https://github.com/AutoViML/Auto_TS/blob/master/updates.md).
<br>

## Citation
If you use Auto_TS in your research project or paper, please use the following format for citations:

> "Seshadri, Ram (2020). GitHub - AutoViML/Auto_TS: enables you to build and deploy multiple time series models using ML and statistical techniques with a single line of code. Source code: https://github.com/AutoViML/Auto_TS"

<h2 id="introduction">Introduction</h2>

Auto_TS (Auto_TimeSeries) enables you to build and select multiple time series models using techniques such as ARIMA, SARIMAX, VAR, decomposable (trend+seasonality+holidays) models, and ensemble machine learning models.

Auto_TimeSeries is an Automated ML library for time series data. Auto_TimeSeries was initially conceived and developed by [Ram Seshadri](https://www.linkedin.com/in/ram-seshadri-nyc-nj/) and was significantly expanded in functionality and scope and upgraded to its present status by [Nikhil Gupta](https://github.com/ngupta23).

auto-ts.Auto_TimeSeries is the main function that you will call with your train data. You can then choose what kind of models you want: stats, ml or Prophet based model. You can also tell it to automatically select the best model based on the scoring parameter you want it to be based on. It will return the best model and a dictionary containing predictions for the number of forecast_periods you mentioned (default=2).

## Install

```bash
pip install auto-ts
```

Use `pip3 install auto-ts` if the above doesn’t work

```bash
pip install git+https://github.com/AutoViML/Auto_TS.git
```

### Installing on Colab 
If you are using Colab or Kaggle kernel and want to install auto_ts, please use the following steps (otherwise you will get an error!):

```
!pip install auto-ts --no-deps --ignore-installed
!pip install 'fsspec>=0.3.3'
!pip install statsmodels --upgrade
!pip install pmdarima
```

![auto_ts_colab](images/install_auto_ts.png)

### Installing on Windows

Windows users may experience difficulties with the Prophet and pystan dependency installations.  Because of this, we recommend installing Prophet using instructions from the [Prophet documentation page](https://facebook.github.io/prophet/docs/installation.html) prior to installing auto-ts.  For Anaconda users, this can be accomplished via:
```bash
conda install -c conda-forge prophet
pip install auto-ts
```

<h2 id="usage">Usage</h2>

### First you need to import auto_timeseries from auto_ts library:<br>

```py
from auto_ts import auto_timeseries
```

### Second, Initialize an auto_timeseries model object which will hold all your parameters:

```py
model = auto_timeseries(
    score_type='rmse',
    time_interval='Month',
    non_seasonal_pdq=None, seasonality=False,
    seasonal_period=12,
    model_type=['Prophet'],
    verbose=2,
)
```

#### Here are how the input parameters defined:

- **score_type (default='rmse')**: The metric used for scoring the models. Type is string.
Currently only the following two types are supported:
  1. "rmse": Root Mean Squared Error (RMSE)
  1. "normalized_rmse": Ratio of RMSE to the standard deviation of actuals
- **time_interval (default is None)**: Used to indicate the frequency at which the data is collected.
This is used for two purposes (1) in building the Prophet model and (2) used to impute the seasonal period for SARIMAX in case it is not provided by the user (None). Type is String. We use the following [pandas date range frequency](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases) aliases that Prophet uses to make the prediction dataframe. <p>Hence, please note that these are the list of allowed aliases for frequency:
          `['B','C','D','W','M','SM','BM','CBM',
         'MS','SMS','BMS','CBMS','Q','BQ','QS','BQS',
         'A,Y','BA,BY','AS,YS','BAS,BYS','BH',
         'H','T,min','S','L,ms','U,us','N']`
For a start, you can test the following codes for your data and see how the results are (or you can leave it as None and auto_timeseries will try and impute it for you):
  - `'MS', 'M', 'SM', 'BM', 'CBM', 'SMS', 'BMS'` for monthly frequency data
  - `'D', 'B', 'C'` for daily frequency data 
  - `'W'` for weekly frequency data 
  - `'Q', 'BQ', 'QS', 'BQS'` for quarterly frequency data 
  - `'A,Y', 'BA,BY', 'AS,YS', 'BAS,YAS'` for yearly frequency data 
  - `'BH', 'H', 'h'` for hourly frequency data 
  - `'T,min'` for minute frequency data 
  - `'S', 'L,milliseconds', 'U,microseconds', 'N,nanoseconds'` for second frequency data
- **non_seasonal_pdq (default = (3,1,3))**: Indicates the maximum value of (p, d, q) to be used in the search for statistical ARIMA models.
If None, then the following values are assumed `max_p = 3, max_d = 1, max_q = 3`. Type is Tuple.
- **seasonality (default=False)**: Used in the building of the SARIMAX model only at this time. True or False. Type is bool.
- **seasonal_period (default is None)**: Indicates the seasonal period in your data. This depends on the peak (or valley) period that occurs regularly in your data.
Used in the building of the SARIMAX model only at this time.
There is no impact of this argument if seasonality is set to False
If None, the program will try to infer this from the time_interval (frequency) of the data
We assume the following as defaults but feel free to change them.
    1. If frequency is Monthly, then seasonal_period is assumed to be 12
    1. If frequency is Daily, then seasonal_period is assumed to be 30 (but it could be 7)
    1. If frequency is Weekly, then seasonal_period is assumed to be 52
    1. If frequency is Quarterly, then seasonal_period is assumed to be 4
    1. If frequency is Yearly, then seasonal_period is assumed to be 1
    1. If frequency is Hourly, then seasonal_period is assumed to be 24
    1. If frequency is Minutes, then seasonal_period is assumed to be 60
    1. If frequency is Seconds, then seasonal_period is assumed to be 60
Type is integer
- **conf_int (default=0.95)**: Confidence Interval for building the Prophet model. Default: 0.95. Type is float.
- **model_type (default: 'stats'**: The type(s) of model to build. Default to building only statistical models. If a list is provided, then only those models will be built. Can be a string or a list of models. Allowed values are:
`'best', 'prophet', 'stats', 'ARIMA', 'SARIMAX', 'VAR', 'ML'`.
  - `"prophet"` will build a model using Prophet -> this means you must have Prophet installed
  - `"stats"` will build statsmodels based ARIMA, SARIMAX and VAR models
  - `"ML"` will build a machine learning model using Random Forests provided explanatory vars are given
  - `"best"` will try to build all models and pick the best one
- **verbose (default=0)**: Indicates the verbosity of printing. Type is integer.

WARNING: "best" might take some time for large data sets. We recommend that you
choose a small sample from your data set before attempting to run entire data.

### The next step after defining the model object is to fit it with some real data:

```py
model.fit(
    traindata=train_data,
    ts_column=ts_column,
    target=target,
    cv=5,
    sep=","
)
```

Here are how the parameters defined:
- **traindata (required)**: It can be either a dataframe or a file. You must give the name of the file along with its data path in case if a file. It also accepts a pandas dataframe in case you already have a dataframe loaded in your notebook.
- **ts_column (required)**: name of the datetime column in your dataset (it could be a name of column or index number in the columns index).
- **target (required)**: name of the column you   are trying to predict. Target could also be the only column in your dataset.
- **cv (default=5)**: You can enter any integer for the number of folds you want in your cross validation data set.
- **sep (default=",")**: Sep is the separator in your traindata file. If your separator is ",", "\t", ";", make sure you enter it here. If not, it is ignored.

### The next step after training the model object is to make some predictions with test data:

```py
predictions = model.predict(
    testdata = ...,  # can be either a dataframe or an integer standing for the forecast_period,
    model = 'best'  # or any other string that stands for the trained model
)  
```

Here are how the parameters are defined. You can choose to send either testdata in the form of a dataframe or send in an integer to decide how many periods you want to forecast. You need only
- **testdata (required)**: It can be either a dataframe containing test data or you can use an integer standing for the forecast_period (you want).
- **model (optional, default = 'best')**: The name of the model you want to use among the many different models you have trained. Remember that the default is the best model. But you can choose any model that you want to forecast with. Type is String.

<h2 id="requirements">Requirements</h2>
dask, scikit-learn, prophet, statsmodels, pmdarima, XGBoost

<h2 id="license">License:</h2>
Apache License 2.0

<h2 id="Tips">Tips</h2>

- We recommend that you choose a small sample from your data set before attempting to run entire data. and the evaluation metric, so it can select the best model. Currently models within “stats” are compared using AIC and BIC. However, models across different types are compared using RMSE. The results of models are shown using RMSE and Normalized RMSE (ratio of RMSE to the standard deviation of actuals).
- You must clean the data and not have any missing values. Make sure the target variable is numeric, otherwise, it won’t run. If there is more than one target variable in your data set, just specify only one for now, and if you know the time interval that is in your data, you can specify it. Otherwise it auto-ts will try to infer the time interval on its own.
- If you give Auto_Timeseries a different time interval than what the data has, it will automatically resample the data to the given time interval and use the mean of the target for the resampled period.
- Notice that except for filename and ts_column input arguments, which are required, all other arguments are optional.
- Note that optionally you can give a separator for the data in your file. Default is comma (",").
- “time_interval” options are any codes that you can find in this page below.
[Pandas date-range frequency aliases](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#timeseries-offset-aliases)
- Optionally, you can give seasonal_period as any integer that measures the seasonality in the data. If not given, seasonal_period is assumed automatically as follows: 
  - Months = 12,
  - Days = 30,
  - Weeks = 52,
  - Qtr = 4,
  - Year = 1,
  - Hours = 24,
  - Minutes = 60 and 
  - Seconds = 60.
- If you want to give your own non-seasonal order, please input it as non_seasonal_pdq and for seasonal order, use seasonal_PDQ as the input. Use tuples. For example, `seasonal_PDQ = (2,1,2)` and `non_seasonal_pdq = (0,0,3)`. It will accept only tuples. The default is None and Auto_Timeseries will automatically search for the best p,d,q (for Non Seasonal) and P, D, Q (for Seasonal) orders by searching for all parameters from 0 to 12 for each value of p,d,q and 0-3 for each P, Q and 0-1 for D.

<h2 id="disclaimer">DISCLAIMER:</h2>

This is not an Officially supported Google project.

<h2 id="copyright">Copyright</h2>

© Google
