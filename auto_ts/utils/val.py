import numpy as np # type: ignore
import pandas as pd # type: ignore
import matplotlib.pyplot as plt # type: ignore
import seaborn as sns # type: ignore
# This gives an error when running from a python script.
# Maybe, this should be set in the jupyter notebook directly.
# get_ipython().magic('matplotlib inline')
sns.set(style="white", color_codes=True)

from sklearn.model_selection import TimeSeriesSplit # type: ignore
from sklearn.model_selection import GridSearchCV # type: ignore

#########################################################
def cross_validation_time_series(model, df, preds, target,n_times=10,verbose=0):
    """
    This splits a time series data frame "n" times as specified in the input (default=10)
    Initially it will start with a certain number of rows in train but it will gradually
    increase train size in steps (which it will calculate automatically) while the
    number of test rows will remain the same (though their content will vary).
    This utility is based on sklearn's time_series_split()
    """
    if n_times > 10:
        print('More than 10 splits is not recommended. Setting n_times to 10')
        n_times = 10
    splits = TimeSeriesSplit(n_splits=n_times)
    index = 0
    X = df[preds].values
    y = df[target].values
    non_df = {}
    rmse_list = []
    for train_index, test_index in splits.split(X):
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]
        if verbose == 1:
            print('Iteration %d: Total Observations = %d' %(index,len(X_train)+len(X_test)))
            print('    Training Index %d Observations: %s' %(len(train_index),train_index))
            print('    Testing Index %d Observations: %s' %(len(test_index),test_index))
        model.fit(X_train, y_train)
        # TODO: Check print_rmse is not defined or loaded
        rmse = print_rmse(y_test, model.predict(X_test))
        rmse_list.append(rmse)
        norm_rmse = rmse/y_test.std()
        print('     Split %d: Normalized RMSE = %0.2f' %(norm_rmse))
        non_df[index] = norm_rmse
        index += 1
    non_df = pd.Series(non_df)
    non_df.plot()
    ave_norm_rmse = np.mean(rmse_list)/y.std()
    print('Normalized RMSE over  entire data after %d splits = 0.2f' %(index,ave_norm_rmse))
    return ave_norm_rmse
##########################################################
def rolling_validation_time_series(model, df, preds, target,train_size=0,
                                                    test_size=0, verbose=0):
    """
    This utility uses a Walk Forward or Rolling Period time series cross validation method.
    Initially it will start with a minimum number of observations to train the model.
    It then gradually increases the train size in steps (which it will calculate automatically)
    while fixing the number of test rows the same (though their content will vary).
    Once the train+test series exceeds the number of rows in data set, it stops.
    It does  not use SKLearn's Time Series Split. You need to provide the initial sizes
    of train and test and it will take care of the rest.
    """
    df = df[:]
    index = 0
    X = df[preds].values
    y = df[target].values
    non_df = {}
    # rmse_list = []  # # TODO: Unused (check)
    if train_size == 0:
        train_size = np.int(np.ceil(len(y)/2))
    if test_size == 0:
        test_size = np.int(np.ceil(len(y)/4))
    # step_size = np.int(np.ceil(test_size/10))  # TODO: Unused (check)
    n_records = len(X)
    ### This contains the start point of test size for each K-Fold in time series
    test_list = np.floor(np.linspace(train_size,n_records-1,5)).tolist()
    for i in range(4):
        train_size = np.int(test_list[i])
        test_size = np.int(test_list[i+1] - test_list[i])
        X_train, X_test = X[:train_size],X[train_size:train_size+test_size]
        y_train, y_test = y[:train_size],y[train_size:train_size+test_size]
        model.fit(X_train, y_train)
        if i == 0:
            ### Since both start and end points are included, you have to subtract 1 from index in this
            df.loc[:train_size-1,'predictions'] = y[:train_size]
            df.loc[train_size:train_size+test_size-1,'predictions'] = model.predict(X_test)
        elif i == 3:
            test_size = np.int(len(X) - train_size)
            X_train, X_test = X[:train_size],X[train_size:train_size+test_size]
            y_train, y_test = y[:train_size],y[train_size:train_size+test_size]
            df.loc[train_size:train_size+test_size,'predictions'] = model.predict(X_test)
        else:
            df.loc[train_size:train_size+test_size-1,'predictions'] = model.predict(X_test)
        if len(y_train) + len(y_test) >= df.shape[0]:
            if verbose:
                print('Iteration %d: Observations:%d' %(index+1,len(X_train)+len(X_test)))
                print('    Train Size=%d, Test Size=%d' %(len(y_train),len(y_test)))
            # TODO:
            rmse = print_rmse(y_test, model.predict(X_test))
            norm_rmse = rmse/y_test.std()
            non_df[i] = rmse
            if verbose:
                print('Normalized RMSE = %0.2f' %norm_rmse)
            non_df = pd.Series(non_df)
            weighted_ave_rmse = np.average(non_df.values,weights=non_df.index,axis=0)
            print('\nWeighted Average of RMSE (%d iterations) = %0.2f\n    Normalized Wtd Aver. RMSE (using std dev) = %0.2f'
                                %(index+1, weighted_ave_rmse,weighted_ave_rmse/y[:].std()))
            #############################
            if verbose == 1 or verbose == 2:
                fig, ax1 = plt.subplots(nrows=1,ncols=1,figsize=(12,8))
                ax1.plot(df[target],label='In-Sample Data', linestyle='-')
                ax1.plot(df['predictions'],'g',alpha=0.6,label='Rolling Forecast')
                ax1.set_xlabel('Time')
                ax1.set_ylabel('Values')
                ax1.legend(loc='best')
            return weighted_ave_rmse, weighted_ave_rmse/y[:].std(), df
        else:
            if verbose:
                print('Iteration %d: Observations:%d' %(index+1,len(X_train)+len(X_test)))
                print('    Train Size=%d, Test Size=%d' %(len(y_train),len(y_test)))
            # TODO: Check print_rmse is not defined or loaded
            rmse = print_rmse(y_test, model.predict(X_test))
            norm_rmse = rmse/y_test.std()
            non_df[i] = rmse
            if verbose:
                print('Normalized RMSE = %0.2f' %norm_rmse)
            index += 1


###################################################
# Re-run the above statistical tests, and more. To be used when selecting viable models.
def ts_model_validation(model_results):
    """
    Once you have built a time series model, how to validate it. This utility attempts to.
    This is only done on SARIMAX models from statsmodels. Don't try it on other models.
    The input is model_results which is the variable assigned to the model.fit() method.
    """
    het_method='breakvar'
    norm_method='jarquebera'
    sercor_method='ljungbox'
    ########################
    (het_stat, het_p) = model_results.test_heteroskedasticity(het_method)[0]
    norm_stat, norm_p, skew, kurtosis = model_results.test_normality(norm_method)[0]
    sercor_stat, sercor_p = model_results.test_serial_correlation(method=sercor_method)[0]
    sercor_stat = sercor_stat[-1] # last number for the largest lag
    sercor_p = sercor_p[-1] # last number for the largest lag

    # Run Durbin-Watson test on the standardized residuals.
    # The statistic is approximately equal to 2*(1-r), where r is the sample autocorrelation of the residuals.
    # Thus, for r == 0, indicating no serial correlation, the test statistic equals 2.
    # This statistic will always be between 0 and 4. The closer to 0 the statistic,
    # the more evidence for positive serial correlation. The closer to 4,
    # the more evidence for negative serial correlation.
    # Essentially, below 1 or above 3 is bad.

    # TODO: Checdk statsmodel is not loaded as sm.
    dw = sm.stats.stattools.durbin_watson(model_results.filter_results.standardized_forecasts_error[0, model_results.loglikelihood_burn:])

    # check whether roots are outside the unit circle (we want them to be);
    # will be True when AR is not used (i.e., AR order = 0)
    arroots_outside_unit_circle = np.all(np.abs(model_results.arroots) > 1)
    # will be True when MA is not used (i.e., MA order = 0)
    maroots_outside_unit_circle = np.all(np.abs(model_results.maroots) > 1)

    print('Test heteroskedasticity of residuals ({}): stat={:.3f}, p={:.3f}'.format(het_method, het_stat, het_p));
    print('\nTest normality of residuals ({}): stat={:.3f}, p={:.3f}'.format(norm_method, norm_stat, norm_p));
    print('\nTest serial correlation of residuals ({}): stat={:.3f}, p={:.3f}'.format(sercor_method, sercor_stat, sercor_p));
    print('\nDurbin-Watson test on residuals: d={:.2f}\n\t(NB: 2 means no serial correlation, 0=pos, 4=neg)'.format(dw))
    print('\nTest for all AR roots outside unit circle (>1): {}'.format(arroots_outside_unit_circle))
    print('\nTest for all MA roots outside unit circle (>1): {}'.format(maroots_outside_unit_circle))
############################################################################################################
def quick_ts_plot(y_true, y_pred, modelname='Prophet'):
    fig,ax = plt.subplots(figsize=(15,7))
    labels = ['actual','forecast']
    y_true.plot(ax=ax,)
    y_pred.plot(ax=ax,)
    ax.legend(labels)
    plt.title('%s: Actual vs Forecast in expanding (training) window Cross Validation' %modelname, fontsize=20);
##############################################################################################
