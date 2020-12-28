import numpy as np # type: ignore
import pandas as pd  # type: ignore
import itertools
import operator
# helper functions
from ..utils import print_static_rmse, print_dynamic_rmse


#########################################################
def build_pyflux_model(df, target, ar=3, ma=3,integ=1, forecast_period=2,
                       fitmethod='MLE', nsims=100, score_type='rmse', verbose=0):
    """
    Build a quick pyflux model with default parameters for AR, MA and I terms in ARIMA.
    You can build a rolling forecast using the rolling_forecast parameter.
    PyFlux is a fiendishly complicated program with very poor documentation.
    I had to dig deep into the API to figure these things out especially the
    """
    try:
        # imported pyflux pkg
        import pyflux as pf  # type: ignore
    except:
        print('Pyflux is not installed - hence not running PyFlux model')
        return 'error','error','error','error'
    ts_df = df[:]
    ##############################################################################
    ts_train = ts_df[:-forecast_period]
    ts_test = ts_df[-forecast_period:]
    if verbose == 1:
        print('Data Set split into train %s and test %s for Cross Validation Purposes'
              % (ts_train.shape, ts_test.shape))
    #####################################################################################################
    if integ > 1:
        print('    Setting "integration"=1 since differenced predictions > 1 are difficult to interpret')
        integ = 1
    if fitmethod == 'M-H':
        print('    Assuming number of simulations = %d' % nsims)
    ####################################################################################################
    ###### define p,d,q parameters here ####################
    p = range(0, ar+1)
    q = range(0, ma+1)
    d = range(0, integ+1)  ### dont do much more than 1 differencing in PyFlux models since its hard to undo
    #### Generate all different combinations of p,d,q triplets ######
    pdq = list(itertools.product(p, d, q))
    eval_metrics = {}
    print('Cycling through various (p,d,q) parameters')
    for param in pdq:
        if verbose == 1:
            print('.', end="")
        model = pf.ARIMA(data=ts_train, ar=param[0], integ=param[1], ma=param[2], target=target)
        try:
            if fitmethod == 'MLE':
                x = model.fit()
            elif fitmethod == 'M-H':
                x = model.fit('M-H', nsims=nsims)
        except:
            x = model.fit('MLE')
        mu, actuals = model._model(model.latent_variables.get_z_values())
        predicted = model.link(mu)
        rmse, norm_rmse = print_static_rmse(actuals,predicted)
        if score_type == 'rmse':
            eval_metrics[param] = rmse
        else:
            eval_metrics[param] = norm_rmse
    bestpdq = min(eval_metrics.items(), key=operator.itemgetter(1))[0]
    print('\nBest Params Selected (based on %s): %s' % (score_type, bestpdq))
    bestmodel = pf.ARIMA(data=ts_train, ar=bestpdq[0], integ=bestpdq[1], ma=bestpdq[2], target=target)
    x = bestmodel.fit()
    if verbose == 1:
        bestmodel.plot_fit(figsize=(15, 5))
    #model.plot_predict_is(h=forecast_period,fit_once=False,fit_method=fitmethod)
    if verbose == 1:
        x.summary()
        n = int(0.5*len(df))
        bestmodel.plot_predict(h=forecast_period, past_values=n, intervals=True, figsize=(15, 5))
    forecast_df = bestmodel.predict(forecast_period, intervals=True)
    mu, actuals = bestmodel._model(bestmodel.latent_variables.get_z_values())
    predicted = bestmodel.link(mu)
    print('Dynamic %d-period Forecasts:' % forecast_period)
    if bestpdq[1] == 1:
        mod_target = 'Differenced ' + target
        res = restore_differenced_predictions(ts_test[target].values, forecast_df[mod_target],
                                              ts_train[target][-1:])
        rmse, norm_rmse = print_dynamic_rmse(ts_test[target].values, res, ts_train[target])
    else:
        rmse, norm_rmse = print_dynamic_rmse(ts_test[target].values,forecast_df[target].values, ts_train[target])
    return bestmodel, forecast_df, rmse, norm_rmse


def restore_differenced_predictions(actuals, predicted, start_value, func=None, periods=1, diff_yes=True):
    try:
        restored = pd.Series(index=start_value.index)
        restored.ix[start_value.ix[:periods].index] = start_value.values[:periods]
        rest = restored.ix[predicted.index]
        restored = pd.Series(np.r_[restored, rest], index=np.r_[start_value.index, rest.index])
        restored.ix[predicted.index] = predicted.values
        restored = restored[(periods-1):].cumsum()
        if func:
            restored = eval('np.' + func + '(restored)')
        return restored[periods:]
    except:
        restored = start_value.values+predicted
        if func:
            restored = eval('np.' + func + '(restored)')
        return restored
