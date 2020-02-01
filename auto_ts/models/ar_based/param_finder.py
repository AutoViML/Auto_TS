import numpy as np
import pandas as pd
import itertools
import operator
import copy
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')
sns.set(style="white", color_codes=True)
# imported SARIMAX from statsmodels pkg for find_best_pdq_or_PDQ
from statsmodels.tsa.statespace.sarimax import SARIMAX


def find_lowest_pq(df):
    """
    This is an auto-ARIMA function that iterates through parameters pdq and finds the best
    based on aan eval metric sent in as input.

    This finds the row and column numbers of the lowest or highest value in a dataframe. All it needs is numeric values.
    It will return the row and column together as a string, you will have to split it into two.
    It will also return the lowest value in the dataframe by default but you can change it to "max".
    """
    dicti = {}
    for ma in list(df):
        try:
            dicti[ma + ' ' + df[ma].idxmin()] = df[ma].sort_values()[0]
        except:
            pass
    lowest_bic = min(dicti.items(), key=operator.itemgetter(1))[1]
    lowest_pq = min(dicti.items(), key=operator.itemgetter(1))[0]
    ma_q = int(lowest_pq.split(' ')[0][2:])
    ar_p = int(lowest_pq.split(' ')[1][2:])
    print('    Best AR order p = %d, MA order q = %d, Interim metric = %0.3f' % (ar_p, ma_q, lowest_bic))
    return ar_p, ma_q, lowest_bic


def find_best_pdq_or_PDQ(ts_train, metric, p_max, d_max, q_max, non_seasonal_pdq,
                         seasonal_period, seasonality=False, verbose=0):
    p_min = 0
    d_min = 0
    q_min = 0
    if seasonality:
        ns_p = non_seasonal_pdq[0]
        ns_d = non_seasonal_pdq[1]
        ns_q = non_seasonal_pdq[2]
    # Initialize a DataFrame to store the results
    iteration = 0
    results_dict = {}
    for d_val in range(d_min, d_max+1):
        print('\nDifferencing = %d' % d_val)
        results_bic = pd.DataFrame(index=['AR{}'.format(i) for i in range(p_min, p_max+1)],
                                   columns=['MA{}'.format(i) for i in range(q_min, q_max+1)])
        for p_val, q_val in itertools.product(range(p_min,p_max+1), range(q_min, q_max+1)):
            if p_val == 0 and d_val == 0 and q_val == 0:
                results_bic.loc['AR{}'.format(p_val), 'MA{}'.format(q_val)] = np.nan
                continue
            try:
                if seasonality:
                    #### In order to get forecasts to be in the same value ranges of the
                    #### orig_endogs, you must set the simple_differencing = False and
                    #### the start_params to be the same as ARIMA.
                    #### THat is the only way to ensure that the output of this
                    #### model is comparable to other ARIMA models
                    model = SARIMAX(ts_train, order=(ns_p, ns_d, ns_q),
                                    seasonal_order=(p_val, d_val, q_val, seasonal_period),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False,
                                    simple_differencing=False, trend='ct',
                                    start_params=[0, 0, 0, 1])
                else:
                    model = SARIMAX(ts_train, order=(p_val, d_val, q_val),
                                    enforce_stationarity=False,
                                    enforce_invertibility=False,
                                    simple_differencing=False, trend='ct',
                                    start_params=[0, 0, 0,1]
                                        )
                    results = model.fit()
                    results_bic.loc['AR{}'.format(p_val), 'MA{}'.format(q_val)] = eval('results.' + metric)
                    if iteration % 10 == 0:
                        print('    Iteration %d completed...' % iteration)
                        iteration += 1
                    elif iteration >= 100:
                        print('    Ending Iterations at %d' % iteration)
                        break
            except:
                iteration += 1
                continue
        results_bic = results_bic[results_bic. columns].astype(float)
        interim_d = d_val
        if results_bic.isnull().all().all():
            print('    D = %d results in an empty ARMA set. Setting Seasonality to False since model might overfit' %d_val)
            #### Set Seasonality to False if this empty condition happens repeatedly ####
            seasonality = False
            continue
        else:
            seasonality = True
        interim_p, interim_q, interim_bic = find_lowest_pq(results_bic)
        if verbose == 1:
            fig, ax = plt.subplots(figsize=(20, 10))
            ax = sns.heatmap(results_bic, mask=results_bic.isnull(), ax=ax, annot=True, fmt='.0f')
            ax.set_title(metric)
        results_dict[str(interim_p)+' '+str(interim_d)+' '+str(interim_q)] = interim_bic
    try:
        best_bic = min(results_dict.items(), key=operator.itemgetter(1))[1]
        best_pdq = min(results_dict.items(), key=operator.itemgetter(1))[0]
        best_p = int(best_pdq.split(' ')[0])
        best_d = int(best_pdq.split(' ')[1])
        best_q = int(best_pdq.split(' ')[2])
    except:
        best_p = copy.deepcopy(p_val)
        best_q = copy.deepcopy(q_val)
        best_d = copy.deepcopy(d_val)
        best_bic = 0
    return best_p, best_d, best_q, best_bic, seasonality
