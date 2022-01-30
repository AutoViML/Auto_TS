from typing import List
import numpy as np
import pandas as pd  # type: ignore
import copy
import pdb
from sklearn.model_selection import TimeSeriesSplit  # type: ignore
import dask
import dask.dataframe as dd

##### This function loads a time series data and sets the index as a time series
def load_ts_data(filename, ts_column, sep, target, dask_xgboost_flag=0):
    """
    This function loads a given filename into a pandas dataframe and sets the
    ts_column as a Time Series index. Note that filename should contain the full
    path to the file.
    Inputs:
        filename: name of file that contains data
        ts_column: name of time series column in data
        sep: separator used as a column separator in datafile
        target: name of the target column to predict
        dask_xgboost_flag: flag that will tell whether to load into dask or pandas dataframe.        
        If dask_xgboost_flag is set to True it returns both a dask as well as pandas DataFrame.
        If dask_xgboost_flag is set to False it returns both of them as pandas DataFrames.

    Outputs:
        dft: dask DataFrame
        filename: pandas DataFrame
    """
    if isinstance(filename, str):
        filename = pd.read_csv(filename, sep=sep, parse_dates=[ts_column])
    ### If filename is not a string, it must be a dataframe and can be loaded
    if dask_xgboost_flag:
        if type(filename) == dask.dataframe.core.DataFrame:
            print('    Since dask_xgboost_flag is True, and input is dask, continuing...')
        else:
            filename = copy.deepcopy(filename)
            print('    Since dask_xgboost_flag is True and input is pandas, reducing memory size of df and loading into dask')
            filename = reduce_mem_usage(filename)
            dft =   dd.from_pandas(filename, npartitions=1)
            print('    Converted pandas dataframe into a Dask dataframe ...' )
    else:
        dft = copy.deepcopy(filename)
        print('    Using given input: pandas dataframe...')
    ##################    L O A D    T E S T   D A T A      ######################
    dft = remove_duplicate_cols_in_dataset(dft)
    #######   Make sure you change it to a date-time index #####
    if dask_xgboost_flag:
        ### if dask exists, you need to change its datetime index also ##
        dft, _ = change_to_datetime_index(dft, ts_column)
    ### you have to change the pandas df also to datetime index ###
    filename, str_format = change_to_datetime_index(filename, ts_column)    
    #preds = [x for x in list(dft) if x not in [target]]
    #dft = dft[[target]+preds]
    return dft, filename, str_format
####################################################################################################################
def load_test_data(filename, ts_column, sep, target, dask_xgboost_flag=0):
    """
    This function loads a given filename into a pandas dataframe and sets the
    ts_column as a Time Series index. Note that filename should contain the full
    path to the file.
    """
    
    if isinstance(filename, str):
        filename = pd.read_csv(filename, sep=sep, index_col=ts_column, parse_dates=True)
        ### If filename is not a string, it must be a dataframe and can be loaded
    else:
        if type(filename) == dask.dataframe.core.DataFrame:
            print('    Since dask_xgboost_flag is True, and input is dask, continuing...')
            ddf =   filename.compute()
            print('    Converted dask dataframe into a pandas dataframe ...' )
            print('    Reducing memory size of df and loading into dask')
            dft = reduce_mem_usage(ddf)
        else:
            dft = copy.deepcopy(filename)
            print('    Using given input: pandas dataframe...')
    ##################    L O A D    T E S T   D A T A      ######################
    dft = remove_duplicate_cols_in_dataset(dft)
    return dft
####################################################################################################################
def remove_duplicate_cols_in_dataset(df):
    df = copy.deepcopy(df)
    cols = df.columns.tolist()
    number_duplicates = df.columns.duplicated().astype(int).sum()
    if  number_duplicates > 0:
        print('Detected %d duplicate columns in dataset. Removing duplicates...' %number_duplicates)
        df = df.loc[:,~df.columns.duplicated()]
    return df
###########################################################################
def change_to_datetime_index(dft, ts_column):
    dft = copy.deepcopy(dft)
    if isinstance(dft, pd.Series) or isinstance(dft, pd.DataFrame):
        try:
            ### If ts_column is not a string column, then set its format to an empty string ##
            str_format = ''
            ############### Check if it has an index or a column with the name of train time series column ####
            
            if ts_column in dft.columns:
                print('    %s column exists in given train data...' %ts_column)
                str_first_value = dft[ts_column].values[0]
                str_values = dft[ts_column].values[:12] ### we want to test a big sample of them 
                if type(str_first_value) == str:
                    ### if it is an object column, convert ts_column into datetime and then set as index
                    str_format = infer_date_time_format(str_values)
                    if str_format:
                        str_format = str_format[0]
                        ts_index = pd.to_datetime(dft.pop(ts_column), format=str_format)
                    else:
                        ts_index = pd.to_datetime(dft.pop(ts_column))
                    dft.index = ts_index
                elif type(str_first_value) == pd.Timestamp or type(str_first_value) == np.datetime64:
                    ### if it is a datetime column, then set it as index
                    ### if it a datetime index, then just set the index as is 
                    ts_index = dft.pop(ts_column)
                    dft.index = ts_index
                elif type(str_first_value) in [np.int8, np.int16, np.int32, np.int64]:
                    ### if it is an integer column, convert ts_column into datetime and then set as index
                    ts_index = pd.to_datetime(dft.pop(ts_column))
                    dft.index = ts_index
                else:
                    print('    Type of time series column %s is float or unknown. Must be string or datetime. Please check input and try again.' %ts_column)
                    return 
            elif ts_column in dft.index.name:
                print('    train time series %s column is the index on test data...' %ts_column)
                ts_index = dft.index
                str_first_value = ts_index[0]
                str_values = ts_index[:12]
                if type(str_first_value) == str:
                    ### if index is in string format, you must infer its datetime string format and then set datetime index
                    str_format = infer_date_time_format(str_values)
                    if str_format:
                        str_format = str_format[0]
                        ts_index = pd.to_datetime(ts_index, format=str_format)
                    else:
                        ts_index = pd.to_datetime(ts_index)
                    dft.index = ts_index
                elif type(ts_index) == pd.core.indexes.datetimes.DatetimeIndex:
                    ### if dft already has a datetime index, leave it as it is
                    pass
                elif type(ts_index) == pd.DatetimeIndex or dft.index.dtype == '<M8[ns]':
                    ### if dft already has a datatime index, leave it as is
                    pass
                elif type(str_first_value) in [np.int8, np.int16, np.int32, np.int64]:
                    ### if it is not a datetime index, then convert it to datetime and set the index
                    ts_index = pd.to_datetime(ts_index)
                    dft.index = ts_index
                else:
                    print('    Type of index is unknown or float. It must be datetime or string. Please check input and try again.')
                    return
            else:
                print(f"    (Error) Cannot find '{ts_column}' (or index) in given data.")
                return None
        except:
            print('    Trying to convert time series column %s into index erroring. Please check input and try again.' %ts_column)
            return 
    elif type(dft) == dask.dataframe.core.DataFrame:
        str_format = ''
        if ts_column in dft.columns:
            print('    %s column exists in dask data frame...' %ts_column)
            str_first_value = dft[ts_column].compute()[0]
            dft.index = dd.to_datetime(dft[ts_column].compute())
            dft = dft.drop(ts_column, axis=1)
        elif ts_column in dft.index.name:
            print('    train index %s is already a time series index. Continuing...' %ts_column)
        else:
            print(f"    (Error) Model to be used for prediction 'ML'. Hence, input df must have a column (or index) called '{ts_column}' corresponding to the original ts_index column passed during training. No predictions will be made.")
            return None
    else:
        print('    Unable to detect type of data. Please check your input and try again')                        
        return
    return dft, str_format
############################################################################################################
def change_to_datetime_index_test(testdata, ts_column, str_format=''):
    testdata = copy.deepcopy(testdata)
    if str_format:
        print('Date_time string format given as %s' %str_format)
    else:
        print('    Alert: No strf_time_format given for %s. Provide strf_time format during "setup" for better results.' %ts_column)
    ##### This is where we change the time index of test data #############    
    
    try:
        if isinstance(testdata, pd.Series) or isinstance(testdata, pd.DataFrame):
            if ts_column in testdata.columns:
                ###### If the str_format is there, set the column as time series index ##
                ts_index = testdata.pop(ts_column)
                if str_format:
                    ts_index = pd.to_datetime(ts_index, format=str_format)
                else:
                    ts_index = pd.to_datetime(ts_index)
                testdata.index = ts_index
            elif ts_column in testdata.index.name:
                ts_index = testdata.index
                ### now set the index to datetime format
                if str_format:
                    ts_index = pd.to_datetime(ts_index, format=str_format)
                else:
                    ts_index = pd.to_datetime(ts_index)
                testdata.index = ts_index
        elif type(testdata) == dask.dataframe.core.DataFrame:
            #### the below tests work for a dask dataframe as well ##
            if ts_column in testdata.columns:
                ####### Now set the index to datetime index and drop the ts_colum #########
                testdata.index = dd.to_datetime(testdata[ts_column].compute())
                testdata = testdata.drop(ts_column, axis=1)
            elif ts_column in testdata.index.name:
                #### the above test works for a dask dataframe as well ##
                ts_index = testdata.index
                if type(testdata.index.compute().values[0]) in [np.datetime64]:
                    ## it is already in date-time index format - do nothing
                    pass
                else:
                    testdata.index = pd.to_datetime(str_values, format=str_format)
            else:
                print("Error: Cannot detect %s either in columns or index. Please check input and try again." %ts_column)
        else:
            print('Unknown type of testdata. Please check input and try again.')
    except:
        print('    converting testdata to datetime index erroring. Please check input and try again.')
    #### this is where we return the testdata and format 
    return testdata, str_format

#################################################################################
def time_series_split(ts_df):
    """
    This utility splits any dataframe sent as a time series split using the sklearn function.
    """
    tscv = TimeSeriesSplit(n_splits=2)
    train_index, test_index = list(tscv.split(ts_df))[1][0], list(tscv.split(ts_df))[1][1]
    ts_train, ts_test = ts_df[ts_df.index.isin(train_index)], ts_df[
                        ts_df.index.isin(test_index)]
    print(ts_train.shape, ts_test.shape)
    return ts_train, ts_test

#################################################################################
def left_subtract(l1,l2):
    lst = []
    for i in l1:
        if i not in l2:
            lst.append(i)
    return lst
#################################################################################
def convert_timeseries_dataframe_to_supervised(df: pd.DataFrame, namevars, target, n_in=1, n_out=0, dropT=True):
    """
    Transform a time series in dataframe format into a supervised learning dataset while
    keeping dataframe intact.
    Returns the transformed pandas DataFrame, the name of the target column and the names of the predictor columns
    Arguments:
        df: A timeseries dataframe that you want to convert to Supervised dataset.
        namevars: columns that you want to lag in the data frame. Other columns will be untouched.
        target: this is the target variable you intend to use in supervised learning
        n_in: Number of lag periods as input (X).
        n_out: Number of future periods (optional) as output for the taget variable (y).
        dropT: Boolean - whether or not to drop columns at time 't'.
        Returns:
        df: This is the transformed data frame with the time series columns laggged.
        Note that the original columns are dropped if you set the 'dropT' argument to True.
        If not, they are preserved.
    This Pandas DataFrame of lagged time series data is immediately available for supervised learning.

    rtype: pd.DataFrame, str, List[str]
    """
    target = copy.deepcopy(target)
    df = copy.deepcopy(df)
    int_vars  = df.select_dtypes(include='integer').columns.tolist()
    # Notice that we will create a sequence of columns from name vars with suffix (t-n,... t-1), etc.
    drops = []
    int_changes = []
    for i in range(n_in, -1, -1):
        if i == 0:
            for var in namevars:
                addname = var + '(t)'
                df = df.rename(columns={var:addname})
                drops.append(addname)
                if var in int_vars:
                    int_changes.append(addname)
        else:
            for var in namevars:
                addname = var + '(t-' + str(i) + ')'
                df[addname] = df[var].shift(i)
                if var in int_vars:
                    int_changes.append(addname)
    ## forecast sequence (t, t+1,... t+n)
    if n_out == 0:
        n_out = False
    for i in range(1, n_out):
        for var in namevars:
            addname = var + '(t+' + str(i) + ')'
            df[addname] = df[var].shift(-i)
    #	drop rows with NaN values
    df = df.dropna()

    ### Make sure that whatever vars came in as integers return back as integers!
    if int_changes:
        ### only do this if there are some changes to implement ###
        df[int_changes] = df[int_changes].astype(np.int64)
    
    #	put it all together
    for each_target in target:
        df = df.rename(columns={each_target+'(t)':each_target})
    if dropT:
        ### If dropT is true, all the "t" series of the target column (in case it is in the namevars)
        ### will be removed if you don't want the target to learn from its "t" values.
        ### Similarly, we will also drop all the "t" series of name_vars if you set dropT to Trueself.
        try:
            drops.remove(target)
        except:
            pass
        df.drop(drops, axis=1, inplace=True)
    preds = [x for x in list(df) if x not in target]
    
    return df, target, preds
##############################################################################################
def find_max_min_value_in_a_dataframe(df, max_min='min'):
    """
    This returns the lowest or highest value in a df and its row value where it can be found.
    Unfortunately, it does not return the column where it is found. So not used much.
    """
    if max_min == 'min':
        return df.loc[:, list(df)].min(axis=1).min(), df.loc[:, list(df)].min(axis=1).idxmin()
    else:
        return df.loc[:, list(df)].max(axis=1).max(), df.loc[:, list(df)].min(axis=1).idxmax()
##############################################################################################
# THIS IS A MORE COMPLEX ALGORITHM THAT CHECKS MORE SPECIFICALLY FOR A DATE AND TIME FIELD
import datetime as dt
from datetime import datetime, date, time

### This tests if a string is date field and returns a date type object if successful and
##### a null list if it is unsuccessful
def is_date(txt):
    fmts = ('%Y-%m-%d', '%d/%m/%Y', '%d-%b-%Y', '%d/%b/%Y', '%b/%d/%Y', '%m/%d/%Y', '%b-%d-%Y', '%m-%d-%Y',
 '%Y/%m/%d', '%m/%d/%y', '%d/%m/%y', '%Y-%b-%d', '%Y-%B-%d', '%d-%m-%y', '%a, %d %b %Y', '%a, %d %b %y',
 '%d %b %Y', '%d %b %y', '%a, %d/%b/%y', '%d-%b-%y', '%m-%d-%y', '%d-%m-%Y', '%b%d%Y', '%d%b%Y',
 '%Y', '%b %d, %Y', '%B %d, %Y', '%B %d %Y', '%b %Y', '%B%Y', '%b %d,%Y')
    parsed=None
    for fmt in fmts:
        try:
            t = dt.datetime.strptime(txt, fmt)
            Year=t.year
            if Year > 2040 or Year < 1900:
                pass
            else:
                parsed = fmt
                return fmt
                break
        except ValueError as err:
            pass
    return parsed



#### This tests if a string is time field and returns a time type object if successful and
##### a null list if it is unsuccessful
def is_time(txt):
    fmts = ('%H:%M:%S.%f','%M:%S.%fZ','%Y-%m-%dT%H:%M:%S.%fZ','%h:%M:%S.%f','%-H:%M:%S.%f',
            '%H:%M','%I:%M','%H:%M:%S','%I:%M:%S','%H:%M:%S %p','%I:%M:%S %p',
           '%H:%M %p','%I:%M %p')
    parsed=None
    for fmt in fmts:
        try:
            t = dt.datetime.strptime(txt, fmt)
            parsed=fmt
            return parsed
            break
        except ValueError as err:
            pass
    return parsed

#### This tests if a string has both date and time in it. Returns a date-time object and null if it is not

def is_date_and_time(txt):
    fmts = ('%d/%m/%Y  %I:%M:%S %p', '%d/%m/%Y %I:%M:%S %p', '%d-%b-%Y %I:%M:%S %p',
 '%d/%b/%Y %I:%M:%S %p', '%b/%d/%Y %I:%M:%S %p', '%Y-%m-%dT%H:%M:%SZ', '%Y-%m-%dT%H:%M:%S.%fZ',
 '%m/%d/%Y %I:%M %p', '%m/%d/%Y %H:%M %p', '%d/%m/%Y  %I:%M:%S', '%d/%m/%Y  %H:%M', '%m/%d/%Y %H:%M',
 '%m/%d/%Y  %H:%M', '%d/%m/%Y  %I:%M', '%d/%m/%Y  %I:%M %p', '%m/%d/%Y  %I:%M', '%d/%b/%Y  %I:%M',
 '%b/%d/%Y  %I:%M', '%m/%d/%Y  %I:%M:%S', '%b-%d-%Y %I:%M:%S %p', '%m-%d-%Y %H:%M:%S %p',
 '%b-%d-%Y %H:%M:%S %p', '%m/%d/%Y %H:%M:%S %p', '%b/%d/%Y %H:%M:%S %p', '%Y-%m-%d %H:%M:%S %Z',
 '%Y-%m-%d %H:%M:%S %Z%z', '%Y-%m-%d %H:%M:%S %z', '%Y/%m/%d %H:%M:%S %Z%z', '%m/%d/%y %H:%M:%S %Z%z',
 '%d/%m/%Y %H:%M:%S %Z%z', '%m/%d/%Y %H:%M:%S %Z%z', '%d/%m/%y %H:%M:%S %Z%z', '%Y-%b-%d %H:%M:%S %Z%z',
 '%Y-%B-%d %H:%M:%S %Z%z', '%d-%b-%Y %H:%M:%S %Z%z', '%d-%m-%y %H:%M:%S %Z%z', '%Y-%m-%d %H:%M',
 '%Y-%b-%d %H:%M', '%a, %d %b %Y %T %z', '%a, %d %b %y %T %z', '%d %b %Y %T %z', '%d %b %y %T %z',
 '%d/%b/%Y %T %z', '%a, %d/%b/%y %T %z', '%d-%b-%Y %T %z', '%d-%b-%y %T %z', '%m-%d-%Y %I:%M %p',
 '%m-%d-%y %I:%M %p', '%m-%d-%Y %I:%M:%S %p', '%d-%m-%Y %H:%M:%S %p', '%m-%d-%y %H:%M:%S %p',
 '%d-%b-%Y %H:%M:%S %p', '%d-%m-%y %H:%M:%S %p', '%d-%b-%y %I:%M:%S %p', '%d-%b-%y %I:%M %p',
 '%d-%b-%Y %I:%M %p', '%d-%m-%Y %H:%M %p', '%d-%m-%y %H:%M %p', '%d/%m/%Y %H:%M:%p', '%d/%m/%Y %H:%M:%S',
 '%Y-%m-%d %H:%M:%S')
    parsed=None
    for fmt in fmts:
        try:
            t = dt.datetime.strptime(txt, fmt)
            parsed=fmt
            return parsed
            break
        except ValueError as err:
            pass
    return parsed

# FIND DATE TIME VARIABLES

# This checks if a field in general is a date or time field

def infer_date_time_format(list_dates):
    """
    This is a generic algorithm that can infer date and time formats by checking repeatedly against a list.
    Make sure you give it a list of datetime formats since there can be many formats in a list.
    You can take the first of the returned list of formats or the majority or whatever you wish.
    # THE DATE FORMATS tested so far by this algorithm are:
        # 19JAN1990
        # JAN191990
        # 19/jan/1990
        # jan/19/1990
        # Jan 19, 1990
        # January 19, 1990
        # Jan 19,1990
        # 01/19/1990
        # 01/19/90
        # 1990
        # Jan 1990
        # January1990 
        # YOU CAN ADD MORE FORMATS above IN THE "fmts" section.
    """
    
    date_time_fmts = []
    try: 
        for each_datetime in list_dates:
            date1 = is_date(each_datetime)
            if date1 and not date1 in date_time_fmts:
                date_time_fmts.append(date1)
            else:
                date2 = is_time(each_datetime)
                if date2 and not date2 in date_time_fmts:
                    date_time_fmts.append(date2)
                else:
                    date3 = is_date_and_time(each_datetime)
                    if date3 and not date3 in date_time_fmts:
                        date_time_fmts.append(date3)
            if not date1 and not date2 and not date3 :
                print('date time format cannot be inferred. Please check input and try again.')
    except:
        print('Error in inferring date time format. Returning...')
    return date_time_fmts
#################################################################################################
def reduce_mem_usage(df):
    """
    #####################################################################
    Greatly indebted to :
    https://www.kaggle.com/arjanso/reducing-dataframe-memory-size-by-65
        for this function to reduce memory usage.
    #####################################################################
    It is a bit slow as it iterate through all the columns of a dataframe and modifies data types
        to reduce memory usage. But it has been shown to reduce memory usage by 65% or so.       
    """
    start_mem = df.memory_usage().sum() / 1024**2
    if type(df) == dask.dataframe.core.DataFrame:
        start_mem = start_mem.compute()
    print('    Caution: We will try to reduce the memory usage of dataframe from {:.2f} MB'.format(start_mem))
    cols = df.columns
    if type(df) == dask.dataframe.core.DataFrame:
        cols = cols.tolist()
    datevars = df.select_dtypes(include=[np.datetime64]).columns.tolist()
    numvars = df.select_dtypes(include='number').columns.tolist()
    for col in cols:
        col_type = df[col].dtype
        if col in datevars:
            pass
        elif col in numvars:
            c_min = df[col].min()
            c_max = df[col].max()
            if type(df) == dask.dataframe.core.DataFrame:
                c_min = c_min.compute()
                c_max = c_max.compute()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    #######  Results after memory usage function ###################
    end_mem = df.memory_usage().sum() / 1024**2
    if type(df) == dask.dataframe.core.DataFrame:
        end_mem = end_mem.compute()
    print('    Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('        decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df
##################################################################################