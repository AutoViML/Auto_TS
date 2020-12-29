from typing import List
import pandas as pd  # type: ignore
import copy
import pdb
from sklearn.model_selection import TimeSeriesSplit  # type: ignore

##### This function loads a time series data and sets the index as a time series
def load_ts_data(filename, ts_column, sep, target):
    """
    This function loads a given filename into a pandas dataframe and sets the
    ts_column as a Time Series index. Note that filename should contain the full
    path to the file.
    """
    if isinstance(filename, str):
        codes_list = ['utf-8', 'iso-8859-1', 'cp1252', 'latin1']
        print('First loading %s and then setting %s as date time index...' % (filename, ts_column))
        for codex in codes_list:
            try:
                df = pd.read_csv(filename, index_col=None, sep=sep, encoding=codex)
                df.index = pd.to_datetime(df.pop(ts_column))
                break
            except:
                print('    Encoder %s or Date time type not working for reading this file...' % codex)
                continue
    else:
        ### If filename is not a string, it must be a dataframe and can be loaded
        dft = copy.deepcopy(filename)
        try:
            dft.index = pd.to_datetime(dft.pop(ts_column))
            preds = [x for x in list(dft) if x not in [target]]
            df = dft[[target]+preds]
        except Exception as e:
            print(e)
            print('Error: Could not convert Time Series column to an index. Please check your input and try again')
            return ''
    return df


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
    df = df[:]
    # Notice that we will create a sequence of columns from name vars with suffix (t-n,... t-1), etc.
    drops = []
    for i in range(n_in, -1, -1):
        if i == 0:
            for var in namevars:
                addname = var + '(t)'
                df.rename(columns={var:addname}, inplace=True)
                drops.append(addname)
        else:
            for var in namevars:
                addname = var + '(t-' + str(i) + ')'
                df[addname] = df[var].shift(i)
    ## forecast sequence (t, t+1,... t+n)
    if n_out == 0:
        n_out = False
    for i in range(1, n_out):
        for var in namevars:
            addname = var + '(t+' + str(i) + ')'
            df[addname] = df[var].shift(-i)
    #	drop rows with NaN values
    df.dropna(inplace=True, axis=0)
    #	put it all together
    target = target+'(t)'
    if dropT:
        ### If dropT is true, all the "t" series of the target column (in case it is in the namevars)
        ### will be removed if you don't want the target to learn from its "t" values.
        ### Similarly, we will also drop all the "t" series of name_vars if you set dropT to Trueself.
        try:
            drops.remove(target)
        except:
            pass
        df.drop(drops, axis=1, inplace=True)
    preds = [x for x in list(df) if x not in [target]]
    return df, target, preds
    ############


def find_max_min_value_in_a_dataframe(df, max_min='min'):
    """
    This returns the lowest or highest value in a df and its row value where it can be found.
    Unfortunately, it does not return the column where it is found. So not used much.
    """
    if max_min == 'min':
        return df.loc[:, list(df)].min(axis=1).min(), df.loc[:, list(df)].min(axis=1).idxmin()
    else:
        return df.loc[:, list(df)].max(axis=1).max(), df.loc[:, list(df)].min(axis=1).idxmax()
