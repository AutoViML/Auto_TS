from typing import Tuple
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt # type: ignore
from sklearn.metrics import mean_absolute_error, mean_squared_error # type: ignore
import pdb

def print_static_rmse(actual: np.array, predicted: np.array, start_from: int=0, verbose: int=0):
    """
    this calculates the ratio of the rmse error to the standard deviation of the actuals.
    This ratio should be below 1 for a model to be considered useful.
    The comparison starts from the row indicated in the "start_from" variable.
    """
    rmse = np.sqrt(mean_squared_error(actual[start_from:], predicted[start_from:]))
    std_dev = actual[start_from:].std()
    if verbose >= 1:
        print('    RMSE = %0.2f' %rmse)
        print('    Std Deviation of Actuals = %0.2f' %(std_dev))
        print('    Normalized RMSE (as pct of std dev) = %0.1f%%' %(rmse*100/std_dev))
    return rmse, rmse/std_dev


def print_dynamic_rmse(actuals: np.array, predicted: np.array, original: np.array, toprint: bool = True):
    """
    This utility calculates rmse between actuals and predicted. However, it does one more.
    Since in dynamic forecast, we need the longer original, it calculates Normalized RMSE
    using the original array's std deviation. That way, the forecast of 2 values does not
    result in a larger Normalized RMSE since the std deviation of 2 values will be v small.
    """
    rmse = np.sqrt(np.mean((actuals - predicted)**2))
    norm_rmse = rmse/original.std()
    if toprint:
        print('    RMSE = {:,.2f}'.format(rmse))
        print('    Std Deviation of actuals = {:,.2f}'.format(actuals.std()))
        print('    Normalized RMSE (as pct of std dev) = %0.0f%%' %(100*norm_rmse))
    return rmse, norm_rmse


def print_normalized_rmse(actuals: np.array, predicted: np.array, start_from: int=0):
    """
    This utility calculates rmse between actuals and predicted. However, it does one more.
    If the original is given, it calculates Normalized RMSE using the original array's std deviation.
    """
    actuals = actuals[start_from:]
    predicted = predicted[start_from:]
    rmse = np.sqrt(np.mean(mean_squared_error(actuals,predicted)))
    norm_rmse = rmse/actuals.std()
    print('RMSE = {:,.2f}'.format(rmse))
    print('Std Deviation of Actuals = {:,.2f}'.format(actuals.std()))
    print('Normalized RMSE = %0.0f%%' %(100*norm_rmse))
    return rmse, norm_rmse


def print_rmse(y: np.array, y_hat: np.array):
    """
    Calculating Root Mean Square Error https://en.wikipedia.org/wiki/Root-mean-square_deviation
    """
    mse = np.mean((y - y_hat)**2)
    return np.sqrt(mse)


def print_mape(y: np.array, y_hat: np.array):
    """
    Calculating Mean Absolute Percent Error https://en.wikipedia.org/wiki/Mean_absolute_percentage_error
    """
    try:
        perc_err = (100*(y - y_hat))/y
        return np.mean(abs(perc_err))
    except:
        return np.nan


def print_ts_model_stats(actuals: np.array, predicted: np.array, title="Model"):
    """
    This program prints and returns MAE, RMSE, MAPE.
    If you like the MAE and RMSE as a percentage of something, just give that number
    in the input as "number_as_percentage" and it will return the MAE and RMSE as a
    ratio of that number. Returns MAE, MAE_as_percentage, and RMSE_as_percentage
    """
    try:
        number_as_percentage = actuals.std()
        if (predicted.index == actuals.index).all():
            dfplot = pd.DataFrame(actuals).join(pd.DataFrame(predicted))
        else:
            dfplot = pd.DataFrame([actuals.values, predicted.values]).T
            dfplot.columns = ['Actual','Forecast']
        dfplot = dfplot.sort_index()
        plt.figure(figsize=(15,8))
        plt.plot(dfplot)
        plt.legend(['original','predicted'])
        plt.title('%s: Actual vs Forecast in expanding (training) Window Cross Validation' %title, fontsize=20)
    except:
        pass
    print('\n-------------------------------------------')
    print('Model Cross Validation Results:')
    print('-------------------------------------------')
    mae = mean_absolute_error(actuals, predicted)
    mse = mean_squared_error(actuals, predicted)
    print('    MAE (Mean Absolute Error = %0.2f' %mae)
    rmse = np.sqrt(mean_squared_error(actuals,predicted))
    print('    MSE (Mean Squared Error = %0.2f' %mse)
    mape = print_mape(actuals, predicted)
    print("    MAPE (Mean Absolute Percent Error) = %0.0f%%" %(mape))
    print("    RMSE (Root Mean Squared Error) = %0.04f" %(rmse))
    # Normalized RMSE print('RMSE = {:,.Of}'.format(rmse))
    print('    Normalized RMSE (MinMax) = %0.0f%%' %(100*rmse/abs(actuals.max()-actuals.min())))
    rmse_asp = (np.sqrt(mean_squared_error(actuals,predicted))/number_as_percentage)*100
    print('    Normalized RMSE (as Std Dev of Actuals)= %0.0f%%' %rmse_asp)
    return rmse, rmse_asp
