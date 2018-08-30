
import os
from time import time
import pandas as pd
import math
from collections import OrderedDict
import pickle
import statsmodels.api as sm
import statsmodels.tsa.api as tsa
import numpy as np
import warnings
from sklearn.svm import SVR

from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
# from matplotlib import pyplot

from pandas import DataFrame
from pandas import Series
from pandas import concat
import sklearn
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.preprocessing import scale
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, mutual_info_regression
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Embedding
from keras.layers import LSTM

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

train_folder = '/Users/jeanclaudelemoyne/work/Data/Unilever/train/20180808/'
results_folder = '/Users/jeanclaudelemoyne/work/Data/Unilever/train/20180808/results/'

ic_orders_csv = train_folder + 'IC_Orders_sorted.csv'
ic_de_orders_csv = train_folder + 'IC_de_Orders_sorted.csv'
ic_orders_by_year_pkl = train_folder + 'ic_orders_by_year.pkl'
arima_results_pkl = train_folder + 'arima_results.pkl'
arima_results_csv = train_folder + 'arima_results.csv'
ds1_sub_ic_orders_pkl = train_folder + 'ds1_sub_ic_orders.pkl'
ic_sales_by_year_pkl = train_folder + 'ic_sales_by_year.pkl'
ssarima_tabulated_results_csv = train_folder + 'ssarima_tabulated_results.csv'
kantar_kmdt_csv = train_folder + 'kantar_kmdt.cvs'
kantar_pc_csv = train_folder + 'kantar_pc.cvs'
kantar_10_features_csv = train_folder + 'kantar_10_features.cvs'
fu_orders_by_year_pkl = train_folder + 'fu_orders_by_year.pkl'
fu_sales_by_year_pkl = train_folder + 'fu_sales_by_year.pkl'
fu_distributions_apg_csv = train_folder + 'fu_distributions_by_apg.csv'
ic_orders_by_week_csv = train_folder + 'ic_orders_by_week.csv'

ssarima_results_csv = results_folder + 'jcl_ssarima_base.csv'
ssarima_with_epos_results_csv = results_folder + 'jcl_ssarima_with_epos.csv'

svr_results_csv = results_folder + 'jcl_svr_base.csv'
lstm_results_csv = results_folder + 'jcl_lstm_base.csv'


# for LSTM
top_10_fu = OrderedDict()
top_10_fu['TEA'] = ['FGB0726', 'FGB0727', 'FGB0737', 'FGB0748', 'FGB0751',
                    'FGB0754', 'FGB1246', 'FGB6596', 'FGB7036', 'FGB7065']
top_10_fu['IC'] = ['IGB0007', 'IGB0040', 'IGB0041', 'IGB0060', 'IGB0222',
                   'IGB0225', 'IGB4680', 'IGB4681', 'IGB4682', 'IGB4842']

lstm_ic_fu = {'IGB0007': 'UGB016',
              'IGB0222': 'UGB016',
              'IGB0225': 'UGB015',
              # 'IGB4680': 'UGB016',
              # 'IGB4681': 'UGB016',
              # 'IGB4682': 'UGB015',
              'IGB4842': 'UGB015'}


def dehyphen_weeknum(ic_orders_csv, ic_de_orders_csv):
    df_orders = pd.read_csv(ic_orders_csv)

    g = open(ic_de_orders_csv, 'w')
    g.write(','.join(df_orders.columns.values) + '\n')
    # print ','.join(df_orders.columns.values)

    for index, row in df_orders.iterrows():
        week = row['weeknum']
        week = week.replace('-', '')
        row['weeknum'] = week
        strrow = [str(x) for x in row]
        # print ','.join(strrow)
        g.write(','.join(strrow) + '\n')
        if index % 1000 == 0:
            print '... ', index

    g.close()


def to_orders_by_year(ic_de_orders_csv, ic_orders_by_year_pkl):
    df = pd.read_csv(ic_de_orders_csv)
    orders_by_year = OrderedDict()
    for index, row in df.iterrows():
        apg = row['APG']
        fu = row['FU']
        weeknum = row['weeknum']
        year = str(weeknum)[:4]
        orders = row['orders']
        if apg not in orders_by_year:
            orders_by_year[apg] = OrderedDict()
        if fu not in orders_by_year[apg]:
            orders_by_year[apg][fu] = OrderedDict()
        if year not in orders_by_year[apg][fu]:
            orders_by_year[apg][fu][year] = OrderedDict()
        orders_by_year[apg][fu][year][weeknum] = orders

    pickle.dump(orders_by_year, open(ic_orders_by_year_pkl, 'wb'))


def make_time_series_df(ts, colname):
    if ts is None:
        return None
    df_ts = pd.DataFrame.from_dict(ts, orient='index', columns=[colname])
    return df_ts


def select_timeseries(apg, fu, orders_by_year):
    if apg not in orders_by_year:
        return None
    if fu not in orders_by_year[apg]:
        return None

    ts = OrderedDict()
    for year in orders_by_year[apg][fu]:
        for weeknum in orders_by_year[apg][fu][year]:
            ts[weeknum] = orders_by_year[apg][fu][year][weeknum]
    return ts


def zero_fill_series(orders_series, sales_series):
    # print '<<<<<<<<< SERIES GAP >>>>>>>'
    # check for gap in index
    orders_index = list(orders_series.index)
    sales_index = list(sales_series.index)
    # print orders_index
    # print sales_index
    which_series = None
    series_name = None
    if len(orders_index) > len(sales_index):
        which_series = sales_series
        series_name = 'sales_series'
        diff = list(set(orders_index).difference(set(sales_index)))
    else:
        which_series = orders_series
        series_name = 'orders_series'
        diff = list(set(sales_index).difference(set(orders_index)))

    print '%s diff: %s\n' % (series_name, diff)

    for index in diff:
        # print '++++++ >>>>>>>>>>>>>>>>>>>>>>>>>>>>> ', index
        which_series.at[index] = 99.0
        which_series.sort_index(inplace=True)
        # print ':::: =========== ', index, ': ', which_series[index]

    # print series_name, ' size: ', which_series.size
    # print '>>>>>>>>> !!! <<<<<<<<<<'
    return series_name, which_series


def series_aligned(series1, series2):
    index1 = list(series1.index)
    index2 = list(series2.index)
    if series1.shape[0] != series2.shape[0]:
        return False
    # print '???????????????? aligned ???????????????'
    # print index1
    # print index2
    seq = True
    for i in range(len(index1)):
        if index1[i] != index2[i]:
            seq = False
            break
    # print '--------------- ????? ------------------'
    return seq


def eg_arima_predictor(df_ts):
    # pyplot.plot(y_true)
    # pyplot.plot(y_hat, color='red')

    X = df_ts.values
    train, test = X[0:-5], X[-5:]
    history = [x for x in train]
    predictions = list()
    # walk-forward validation
    for t in range(len(test)):
        # fit model
        model = ARIMA(history, order=(5,1,5))
        model_fit = model.fit()
        # one step forecast
        yhat = model_fit.forecast()[0]
        # store forecast and ob
        predictions.append(yhat)
        history.append(test[t])
    # evaluate forecasts
    rmse = math.sqrt(mean_squared_error(test, predictions))
    print('Test RMSE: %.3f' % rmse)
    # plot forecasts against actual outcomes
    pyplot.plot(test)
    pyplot.plot(predictions, color='red')
    pyplot.show()


def forward_shift_predict_demand(timeseries, nforward):
    ts = timeseries.tail(timeseries.shape[0] - nforward)
    # print 'Initial shape: ', timeseries.shape
    # print ' /////// =======> Shifted Forward %d periods; shape %s \n' % (nforward, ts.shape)
    # project the rest
    warnings.filterwarnings("ignore")
    mod = sm.tsa.statespace.SARIMAX(endog=ts, exog=None, trend='t', order=(5,1,7),
                                    enforce_stationarity=False, enforce_invertibility=False)
    res = mod.fit(disp=False)
    projection = res.forecast(nforward)
    last_index = ts.tail(1).index.item()
    index = [last_index + x + 1 for x in range(nforward)]
    projection.index = index
    projection = projection.round(decimals=0)
    # print ' ~~~~~~~~ Projected ~~~~~~~~ '
    # print projection
    ts_ext = ts.append(projection)
    # print '___------ APPENDED -------___ ', last_index
    # print ts_ext
    return ts_ext


def arima_predictor_ver1(apg, fu, orders_df_ts, sales_df_ts, params=(5, 1, 7),
                         from_index=201811, foreper=12, forward_shift=5):
    orders_series = orders_df_ts['orders']
    offset = list(orders_series.index).index(from_index)
    ts_size = offset - 1 - forward_shift
    train_orders_series = orders_series.head(ts_size)
    actual_size = orders_series.shape[0] - ts_size
    actual_orders_series = orders_series.tail(actual_size)
    # number of project periods
    # projected = orders_series.shape[0] - train_orders_series.shape[0]
    projected = foreper
    # print '>>>>>> Projected: %d periods  1st actual %d \n' % (projected,
    #                                                            actual_orders_series.head(1).index[0])

    sum = 0.0; zfreq = 0; ndp = 0
    for index in orders_series.index:
        qty = orders_series[index]
        sum = sum + qty
        if qty < 1.0:
            zfreq = zfreq + 1
        else:
            ndp = ndp + 1

    if sum < 1.0 or ndp < 3:
        return None, None, None, None

    train_sales_series = None
    if sales_df_ts is not None:
        sales_series = sales_df_ts['sales']
        train_sales_series = sales_series.head(ts_size)

    orders_weeknum_1 = orders_series.index[0]
    orders_weeknum_last = orders_series.index[len(orders_series.index) - 1]

    if train_sales_series is not None:
        sales_weeknum_1 = sales_series.index[0]
        sales_weeknum_last = sales_series.index[len(sales_series.index) - 1]

        start_weeknum = max(orders_weeknum_1, sales_weeknum_1)
        last_weeknum = min(orders_weeknum_last, sales_weeknum_last)

        # print '=========== Trimmed Week # ==========='
        # print 'start week #: ', start_weeknum
        # print 'last week #: ', last_weeknum

        orders_series = orders_series.loc[start_weeknum:last_weeknum, ]
        sales_series = sales_series.loc[start_weeknum:last_weeknum, ]
        series_name, zero_filled_series = zero_fill_series(orders_series, sales_series)
        if series_name == 'orders_series':
            orders_series = zero_filled_series
        if series_name == 'sales_series':
            sales_series = zero_filled_series

        if from_index not in orders_series.index:
            return None, None, None, None

        offset = list(orders_series.index).index(from_index)
        ts_size = offset - 1 - forward_shift
        train_orders_series = orders_series.head(ts_size)
        actual_size = orders_series.shape[0] - ts_size
        actual_orders_series = orders_series.tail(actual_size)
        train_sales_series = sales_series.head(ts_size)

        # print '~~~~======= train_orders_series: ', train_orders_series.size
        # print '~~~~======= train_sales_series: ', train_sales_series.size
        # print '~~~~======= series aligned: ', series_aligned(train_orders_series, train_sales_series)
        # print '++++======= train_sales_series: ', train_sales_series.size

    if train_sales_series is not None:
        train_sales_series = forward_shift_predict_demand(train_sales_series, forward_shift)

    exog_var = None
    if train_sales_series is not None:
        exog_var = train_sales_series.values
        exog_var_isfinte = np.isfinite(exog_var)
        if (exog_var_isfinte).all():
            exog_greater_than_1 = np.greater(exog_var, np.ones(exog_var.shape[0]))
            if (exog_greater_than_1).all():
                exog_var = np.log(exog_var)
    # exog_var = None

    warnings.filterwarnings("ignore")
    mod = sm.tsa.statespace.SARIMAX(endog=train_orders_series, exog=exog_var, trend='t', order=params,
                                    enforce_stationarity=False, enforce_invertibility=False)
    try:
        res = mod.fit(disp=False)
    except:
        return None, None, None, None

    # print(res.summary())

    predict = res.get_prediction()
    if exog_var is not None:
        exog_pred = exog_var[exog_var.shape[0] - projected - forward_shift:]
        exog_pred = exog_pred.reshape(exog_pred.shape[0], -1)
        future = res.forecast(forward_shift + projected, exog=exog_pred)
    else:
        future = res.forecast(projected)

    # projection = res.predict(start=201823, end=201835)
    # predict_ci = predict.conf_int()

    y_true = train_orders_series
    y_hat = predict.predicted_mean

    last_index = list(y_true.index)[len(y_true) - 1]

    # print '~~~~~~~~ Projections ~~~~~~~'
    # print projection

    for index in y_hat.index:
        if y_hat[index] < 10.0:
            y_hat[index] = 0.0
        # print index , '  ', y_hat[index], ' ?? == ?? ', y_true[index]

    mse = ((y_hat - y_true) ** 2).mean()
    rmse = math.sqrt(mse)
    y = np.array(y_true)
    stdev = np.std(y)
    normalized_rmse = rmse / stdev

    future_index = [from_index + x for x in range(future.shape[0])]
    future.index = future_index
    # future.index = actual_orders_series.index
    for index in future.index:
        if future[index] < 10.0:
            future[index] = 0.0
        if np.isnan(future[index]):
            future[index] = future[index]
        else:
            future[index] = int(future[index])

    df_future = pd.DataFrame()
    df_future['predicted'] = future
    actual_values = actual_orders_series.values
    nav = actual_orders_series.shape[0]
    npv = df_future.shape[0]
    for i in range(nav - npv):
        actual_values = np.delete(actual_values,-1)
    for i in range(npv - nav):
        actual_values = np.append(actual_values, [np.NAN])
    # print ' ??????? ', df_future.shape, ' ??????? ', actual_values.shape
    df_future['actual'] = actual_values

    # print '---------- df_future ---------'
    # print df_future

    print '====== Performance for APG %s FU %s fom index %s ====== ' % (apg, fu, from_index)
    print '\t++++ forward shift: %d forecast period: %d ' % (forward_shift, projected)
    print('Prediction quality: {:.2f} MSE ({:.2f} RMSE)'.format(mse, rmse))
    print 'std dev = ', stdev, '  normalized RMSE = ', normalized_rmse

    if normalized_rmse > 2.0:
        return None,None, None, None

    return rmse, normalized_rmse, y_hat, df_future


def rolling_ssarima_predictor(apg, fu, orders_df_ts, sales_df_ts):
    from_index=201811
    forward_shift = 5
    params=(5, 1, 5)
    df_predict = pd.DataFrame()

    for nweek in range(12):
        rmse, n_rmse, y_hat, df_future = arima_predictor_ver1(apg, fu, orders_df_ts, sales_df_ts,
                                                              params=params,
                                                              from_index=from_index,
                                                              forward_shift=forward_shift)
        if df_future is None:
            continue

        df_append = df_future.head(1)

        df_append.insert(0, 'APG', Series(apg, index=df_append.index))
        df_append.insert(1, 'FU', Series(fu, index=df_append.index))
        df_append['RMSE'] = Series(np.round(rmse, decimals=0), index=df_append.index)
        df_append['N-RMSE'] = Series(np.round(n_rmse, decimals=4), index=df_append.index)

        df_predict = df_predict.append(df_append)

        weeknum = df_future.head(1).index[0]
        predicted_value = df_future.head(1)['predicted']
        original_value = orders_df_ts.loc[weeknum]['orders']
        orders_df_ts.loc[weeknum]['orders'] = predicted_value

        from_index = from_index + 1

    print '\n~~~~~ forward prediction ~~~~~~~~~~'
    print df_predict

    return df_predict


def arima_predictor_ver2(df_ts):
    series = df_ts['orders']
    sarimax_mod = tsa.SARIMAX(endog=series, order=(2,1,0), seasonal_order=(1,1,0,12))
    sarimax_res = sarimax_mod.fit()
    sarimax_res.summary()

    predict, cov, ci, idx = sarimax_res.predict(alpha=0.05, start=0, end=len(series))

    # show forecast
    print predict

    # show problematic value in forecast
    print predict[0][12]


def back_shift(series, n):
    Z = series.tolist()
    Z_n = Z[n:]
    return Z_n


def ts_diff(series, n):
    Z = series.tolist()
    Z_n = back_shift(series, 1)
    diff = [0]
    for i in range(len(Z_n)):
        diff = diff + [Z[i] - Z_n[i]]
    return diff


'''
        LSTM block of code
        
'''


# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)


# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
    df = DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns.append(df)
    df = concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df


# scale train and test data to [-1, 1]
def scale(train, test):
    # fit scaler
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train)
    # feat_transform train
    train = train.reshape(train.shape[0], train.shape[1])
    train_scaled = scaler.transform(train)
    # feat_transform test
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled


# fit an LSTM network to training data
def fit_lstm(train, batch_size, nb_epoch, neurons):
    X, y = train[:, 0:-1], train[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(nb_epoch):
        if i % 25 == 0:
            print ' ... epoch #: ', i
        model.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
        model.reset_states()
    return model


# make a one-step forecast
def forecast_lstm(model, batch_size, X):
    X = X.reshape(1, 1, len(X))
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0,0]


# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
    new_row = [x for x in X] + [value]
    array = np.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]


# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]


def lstm_predictor(apg, fu, df_ts, from_index=201811):

    series = df_ts['orders']
    offset = list(series.index).index(from_index)
    foreper = series.shape[0] - offset
    raw_values = series.values

    diff_values = difference(raw_values, 1)
    # feat_transform data to be supervised learning
    supervised = timeseries_to_supervised(diff_values, 1)
    supervised_values = supervised.values
    # split data into train and test-sets
    train, test = supervised_values[0:-foreper], supervised_values[-foreper:]

    # feat_transform the scale of the data
    scaler, train_scaled, test_scaled = scale(train, test)
    predindex = series.index[train.shape[0] + 1]

    # fit the model
    print '... fit lst model ...'
    warnings.filterwarnings("ignore")
    batch_size = 1; nb_epoch = 1000
    lstm_model = fit_lstm(train_scaled, batch_size, nb_epoch, 4)
    # forecast the entire training dataset to build up state for forecasting
    train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
    lstm_model.predict(train_reshaped, batch_size=1)

    # walk-forward validation on the test data
    df_predict = pd.DataFrame(columns=['APG', 'FU', 'predicted', 'actual', 'RMSE', 'N-RMSE'],
                              index=range(from_index, from_index + foreper))
    predictions = list()
    for i in range(len(test_scaled)):
        # make one-step forecast
        X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
        yhat = forecast_lstm(lstm_model, 1, X)
        # invert scaling
        yhat = invert_scale(scaler, X, yhat)
        # invert differencing
        yhat = inverse_difference(raw_values, yhat, len(test_scaled)+1-i)
        # store forecast
        predictions.append(yhat)
        expected = max(0, int(raw_values[len(train) + i + 1]))
        yhat = max(0, int(yhat))
        print('%d\tweek=%d\tPredictedv = %d\tExpected = %d' % (from_index, i+1, yhat, expected))
        df_predict.loc[from_index]['APG'] = apg
        df_predict.loc[from_index]['FU'] = fu
        df_predict.loc[from_index]['predicted'] = yhat
        df_predict.loc[from_index]['actual'] = expected
        from_index = from_index + 1

    # report performance
    print '========= LSTM Predictive Quality ========'
    rmse = math.sqrt(mean_squared_error(raw_values[-foreper:], predictions))
    stdev = np.std(raw_values[-foreper:])
    nrmse = rmse / stdev
    print('Test RMSE: %.3f\tN-RMSEL %.3f' % (rmse, nrmse))

    # line plot of observed vs predicted
    # pyplot.plot(raw_values[-12:])
    # pyplot.plot(predictions)
    # pyplot.show()

    for index in df_predict.index:
        df_predict.loc[index]['RMSE'] = np.round(rmse, decimals=4)
        df_predict.loc[index]['N-RMSE'] = np.round(nrmse, decimals=4)

    print '~~~~~~~~ df_predict ~~~~~~~~'
    print df_predict

    return df_predict


def lstm_batch_control_run(ic_orders_by_year_pkl, ic_sales_by_year_pkl, lstm_results='jcl_lstm_results.csv'):

    orders_by_year = pickle.load(open(ic_orders_by_year_pkl, 'rb'))
    sales_by_year = pickle.load(open(ic_sales_by_year_pkl, 'rb'))

    df_all_results = pd.DataFrame()
    t0 = time()
    ii = 0
    for fu in lstm_ic_fu:
        apg = lstm_ic_fu[fu]
        ii = ii + 1
        print '\n{ %d } <<<<<<< APG: %s\tFU: %s >>>>>>>' % (ii, apg, fu)

        ts_orders = select_timeseries(apg, fu, orders_by_year)
        ts_sales = select_timeseries(apg, fu, sales_by_year)

        df_ts_orders = make_time_series_df(ts_orders, 'orders')
        df_ts_sales = make_time_series_df(ts_sales, 'sales')
        df_ts_sales = filter_ts_sales(df_ts_sales)

        df_predict = lstm_predictor(apg, fu, df_ts_orders)

        if df_predict is not None:
            df_all_results = df_all_results.append(df_predict)

    delta = time() - t0
    df_all_results.index.name = 'weeknum'
    df_all_results.to_csv(results_folder + lstm_results)

    print '.... time taken: %d : %d' % (delta / 60, delta % 60)
    print df_all_results


'''
        S V R
'''


def svr_predictor(df_ts):
    print '... SVR model ---'
    series = df_ts['orders']

    raw_values = series.values
    diff_values = difference(raw_values, 1)

    max_lag = 5
    ts_len = len(series)
    n_samples, n_features = ts_len - max_lag, max_lag
    y = np.array(series[0: -max_lag])

    AR = []
    for lag in range(1, max_lag + 1):
        if -max_lag + lag == 0:
            AR = AR + [series[lag:]]
            print lag, ' :', -max_lag + lag, ' >>> ', len(series[lag:])
        else:
            print lag, ' :', -max_lag + lag, ' >>> ', len(series[lag: -max_lag + lag])
            AR = AR + [series[lag: -max_lag + lag]]

    Z_1 = ts_diff(series, 1)
    # AR = AR + [Z_1[max_lag:]]
    AR = AR

    X = np.array(AR).T
    print ' - y shape: ', y.shape
    print ' - X shape: ', X.shape
    # print ' ~~~~~~~ DUMP ~~~~~~~'
    # print ' ==== y ===='
    # print y
    # print ' ==== X ===='
    # print X
    clf = SVR(C=1.0, epsilon=0.2)
    model = clf.fit(X, y)
    print ' ~~~~~~~ MODEL ~~~~~~~'
    print model
    print ' ~~~~~~~ PREDICT ~~~~~~~'
    predictions = clf.predict(X)
    print predictions
    print '----------- y ----------'
    print y
    print ' ~~~~~~~ PREDICT ~~~~~~~'
    print 'R^2 = ', clf.score(X, y)
    print ' ~~~~~~~ RMSE ~~~~~~~'
    rmse = math.sqrt(mean_squared_error(y, predictions))
    stdev = np.std(y)
    print 'RMSE: ', rmse
    print 'Normalized RMSE: ', rmse / stdev


'''
        Random Forest Regressor
'''


def rfr_predictor(ic_orders_by_year_pkl, ic_orders_by_week_csv):
    orders_by_year = pickle.load(open(ic_orders_by_year_pkl, 'rb'))
    g = open(ic_orders_by_week_csv, 'w')
    ii = 0
    row_dict = OrderedDict()
    for apg in orders_by_year:
        for fu in orders_by_year[apg]:
            ii = ii + 1
            ts_orders = select_timeseries(apg, fu, orders_by_year)
            print '%d - %s x %s: %d' % (ii, apg, fu, len(ts_orders))
            if ii == 1:
                hdr = ['"APG"', '"FU"']
                hdr = hdr + ts_orders.keys()
                hdr = ['"' + str(x) + '"' for x in hdr]
                g.write(','.join(hdr) + '\n')
            row_dict.clear()
            for weeknum in ts_orders:
                row_dict[str(weeknum)] = ts_orders[weeknum]
            row = [apg, fu]
            row = row + row_dict.values()
            row = [str(x) for x in row]
            g.write(','.join(row) + '\n')

    g.close()
    # df_orders_by_week = pd.read_csv(ic_orders_by_week_csv)
    # print df_orders_by_week


def load_ds1_ic_orders(ds1_sub_ic_orders_pkl):
    df_trainset = pickle.load(open(ds1_sub_ic_orders_pkl, 'rb'))
    print 'loaded ', ds1_sub_ic_orders_pkl
    print df_trainset.shape
    df_trainset.columns = ['APG', 'FU', 'weeknum', 'orders']
    print df_trainset.columns.values
    return df_trainset


def benchwork_predictions(orders_by_year, random_fu_list):
    h = open(ssarima_tabulated_results_csv, 'w')
    hdr = 'APG,FU,RMSE,N-RMSE,'
    arima_results = OrderedDict()
    # loop through randomly selected APG x FU
    nattempt = 0; nfitted = 0
    first = True
    for apg in random_fu_list:
        if apg not in arima_results:
            arima_results[apg] = OrderedDict()
        for fu in random_fu_list[apg]:
            ts = select_timeseries(apg, fu, orders_by_year)
            df_ts = make_time_series_df(ts, 'orders')
            print '\n\n[trial %d] +++++++++ %s -- %s +++++++++' % (nattempt, apg, fu)
            nattempt = nattempt + 1
            rmse, normal_rmse, y_hat, future = arima_predictor_ver1(apg, fu, df_ts, None)
            if first:
                first = False
                cols = list(future.index)
                cols = [str(x) for x in cols]
                hdr = hdr + ','.join(cols)
                h.write(hdr + '\n')
            if normal_rmse is not None and normal_rmse < 1.5:
                row = [apg, fu, int(rmse), str(round(normal_rmse, 2))] + list(future.values)
                row = [str(x) for x in row]
                # print '**** ', row
                h.write(','.join(row) + '\n')
                arima_results[apg][fu] = normal_rmse
                nfitted = nfitted + 1

    h.close()

    pickle.dump(arima_results, open(arima_results_pkl, 'wb'))
    print '# attempts to fit model: ', nattempt
    print '# fitted: ', nfitted
    print '%% success: %6.0f' % (float(nfitted) * 100.0 / float(nattempt))
    g = open(arima_results_csv, 'w')
    g.write("APG,FU,NORRMSE\n")
    for apg in arima_results:
        for fu in arima_results[apg]:
            g.write('%s,%s,%s\n' % (apg, fu, arima_results[apg][fu]))
    g.close()


def get_frequencies(qty_by_year):
    kk = 0
    fu_set = OrderedDict()
    apg_set = set()
    qty_freq = OrderedDict()
    for apg in qty_by_year:
        apg_set.add(apg)
        if apg not in fu_set:
            fu_set[apg] = set()
        if apg not in qty_freq:
            qty_freq[apg] = []
        for fu in qty_by_year[apg]:
            fu_set[apg].add(fu)
            nper = 0
            for year in qty_by_year[apg][fu]:
                for weeknum in qty_by_year[apg][fu][year]:
                    kk = kk + 1
                    nper = nper + 1
            qty_freq[apg] = qty_freq[apg] + [nper]

    return kk, apg_set, fu_set, qty_freq


def kantar_decompo(kantar_kmdt_csv):
    df_kantar = pd.read_csv(kantar_kmdt_csv)
    irow, jcol = df_kantar.shape
    print df_kantar.shape
    cols = df_kantar.columns.values
    print '# cols: ', len(cols)
    X = df_kantar.values
    print 'X shape: ', X.shape
    # pca = PCA(n_components=10, svd_solver='full')
    # pca.fit(X)
    data_scaled = pd.DataFrame(preprocessing.scale(df_kantar), columns=df_kantar.columns)
    pca = PCA(n_components=7, svd_solver='full')
    # principalComponents = pca.fit_transform(data_scaled)
    principalComponents = pca.fit_transform(df_kantar)
    # coef = pca.feat_transform(X)
    # pca.fit(df_kantar)
    principals = pd.DataFrame(pca.components_, columns=data_scaled.columns,
                              index=['kantar_pc1', 'kantar_pc2', 'kantar_pc3', 'kantar_pc4',
                                     'kantar_pc5', 'kantar_pc6', 'kantar_pc7'])
    # principals = pd.DataFrame(pca.components_, columns=df_kantar.columns,
    #                           index=['PC-1', 'PC-2', 'PC-3', 'PC-4', 'PC-5', 'PC-6', 'PC-7'])
    # print principals
    principalDf = pd.DataFrame(data=principalComponents,
                               columns=['kantar_pc1', 'kantar_pc2', 'kantar_pc3', 'kantar_pc4',
                                        'kantar_pc5', 'kantar_pc6', 'kantar_pc7'])

    principalDf = principalDf[['kantar_pc1', 'kantar_pc2', 'kantar_pc3']]
    principalDf = principalDf.round(2)
    principalDf.to_csv(kantar_pc_csv)

    print '~~~~~ principle_componants ~~~~~'
    print principalDf
    print principalDf.shape
    print '~~~~~ explained variance ratio ~~~~~'
    print pca.explained_variance_ratio_.shape
    print pca.explained_variance_ratio_
    print '~~~~~ Principal components ~~~~~'
    print pca.components_.shape, type(pca.components_)
    print pca.components_


def kantar_pca_decompo(kantar_kmdt_csv):
    df_kantar = pd.read_csv(kantar_kmdt_csv)
    X = df_kantar.values
    mu = np.mean(X, axis=0)
    print 'X shape: ', X.shape
    X = sklearn.preprocessing.scale(X)
    nComp = 7
    pca = PCA()
    pca_data = pca.fit(X)
    pca_inv = pca.inverse_transform(np.eye(X.shape[0]))

    print pca_inv

    # Xhat = np.dot(pca.feat_transform(X)[:, :nComp], pca.components_[:nComp, :])
    # Xhat += mu
    #
    # print(Xhat[0,])

    # var = pca.explained_variance_ratio_
    # var1 = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
    # print var1


def kantar_select_features(kantar_kmdt_csv):
    df_kantar = pd.read_csv(kantar_kmdt_csv)
    features = df_kantar.columns.values
    X = df_kantar.values
    nrow, ncol = X.shape
    X_train = X[:, 1:]
    print 'X shape: ', X.shape
    print 'X_train shape: ', X_train.shape
    y_train = X[:, 0]
    print 'y_train shape: ', y_train.shape
    y = X[:, 0] + np.random.random(nrow)
    X_new = SelectKBest(mutual_info_regression, k=3).fit_transform(X_train, y_train)
    print X_new.shape
    # print X_new

    model = LinearRegression()
    rfe = RFE(model, 10)
    X_train = X
    periods = np.ones(nrow)
    y_train = periods * 1.05 + np.random.random(nrow)
    fit = rfe.fit(X_train, y_train)
    print("Num Features: %d"% fit.n_features_)
    print("Selected Features: %s"% fit.support_)
    print("Feature Ranking: %s"% fit.ranking_)
    print '# features: ', len(features)
    print 'selection size: ', len(fit.support_)

    # pool = features[1:]
    pool = features
    selected_features = []
    for index in range(len(fit.support_)):
        select = fit.support_[index]
        if select:
            selected_features = selected_features + [pool[index]]
            print ' +++++++++++:: ', pool[index], '  ranking', fit.ranking_[index]
    df_10_featues = df_kantar[selected_features]
    df_10_featues.to_csv(kantar_10_features_csv)


def consolidate_to_FU(qty_by_year_pkl, fu_consolidate_by_year_pkl):
    qty_by_year = pickle.load(open(qty_by_year_pkl, 'rb'))
    fu_by_year = OrderedDict()
    for apg in qty_by_year:
        for fu in qty_by_year[apg]:
            if fu not in fu_by_year:
                fu_by_year[fu] = OrderedDict()
            for year in qty_by_year[apg][fu]:
                if year not in fu_by_year[fu]:
                    fu_by_year[fu][year] = OrderedDict()
                for weeknum in qty_by_year[apg][fu][year]:
                    if weeknum not in fu_by_year[fu][year]:
                        fu_by_year[fu][year][weeknum] = qty_by_year[apg][fu][year][weeknum]
                    else:
                        fu_by_year[fu][year][weeknum] = fu_by_year[fu][year][weeknum] + qty_by_year[apg][fu][year][weeknum]
    pickle.dump(fu_by_year, open(fu_consolidate_by_year_pkl, 'wb'))


def make_FU_APG_distributions(qty_by_year_pkl, fu_distributions_apg_csv):
    qty_by_year = pickle.load(open(qty_by_year_pkl, 'rb'))
    fu_distrib = OrderedDict()
    apg_list = []
    for apg in qty_by_year:
        apg_list = apg_list + [apg]
        for fu in qty_by_year[apg]:
            if fu not in fu_distrib:
                fu_distrib[fu] = OrderedDict()
            for year in qty_by_year[apg][fu]:
                for weeknum in qty_by_year[apg][fu][year]:
                    if weeknum not in fu_distrib[fu]:
                        fu_distrib[fu][weeknum] = OrderedDict()
                    if apg not in fu_distrib[fu][weeknum]:
                        fu_distrib[fu][weeknum][apg] = OrderedDict()
                    fu_distrib[fu][weeknum][apg] = qty_by_year[apg][fu][year][weeknum]

    apg_list = sorted(qty_by_year.keys())
    g = open(fu_distributions_apg_csv, 'w')
    hdr = ['FU', 'WEEK'] + apg_list
    g.write(','.join(hdr) + '\n')
    for fu in fu_distrib:
        for weeknum in fu_distrib[fu]:
            row = [fu, weeknum]
            apg_qty = OrderedDict()
            for apg in fu_distrib[fu][weeknum]:
                apg_qty[apg] = fu_distrib[fu][weeknum][apg]
            dist = []
            for apg in apg_list:
                if apg in apg_qty:
                    orders = apg_qty[apg]
                    dist = dist + [orders]
                else:
                    dist = dist + [0.0]
            sum = 0.0
            for qty in dist:
                # print 'qty: ', qty
                sum = sum + qty
            if sum > 0:
                dist = [ np.round(x / sum, decimals=4) for x in dist]
            row = row + dist
            row = [str(x) for x in row]
            g.write(','.join(row) + '\n')
    g.close()


def count_apg_fu_orders(ic_orders_by_year_pkl, ic_sales_by_year_pkl):
    orders_by_year = pickle.load(open(ic_orders_by_year_pkl, 'rb'))
    sales_by_year = pickle.load(open(ic_sales_by_year_pkl, 'rb'))
    n_apg_fu_orders = 0
    counter = OrderedDict()
    lstm_fu = top_10_fu['IC']
    lstm_apg = OrderedDict()
    for apg in orders_by_year:
        if apg not in counter:
            counter[apg] = 0
        for fu in orders_by_year[apg]:
            if fu in lstm_fu:
                if apg not in lstm_apg:
                    lstm_apg[fu] = set()
                lstm_apg[fu] = apg
            n_apg_fu_orders = n_apg_fu_orders + 1
            counter[apg] = counter[apg] + 1
    print 'APG x FU orders combinations: ', n_apg_fu_orders

    n_apg_fu_sales = 0
    n_acceptable_ts = 0
    for apg in sales_by_year:
        for fu in sales_by_year[apg]:
            n_apg_fu_sales = n_apg_fu_sales + 1
            ts_size = 0
            for year in sales_by_year[apg][fu]:
                for weeknum in sales_by_year[apg][fu][year]:
                    ts_size = ts_size + 1
            if ts_size > 120:
                n_acceptable_ts = n_acceptable_ts + 1

    print 'APG x FU sales combinations: ', n_apg_fu_sales
    print ' acceptable sales Timeseries: ', n_acceptable_ts
    print ' ~~~~~~~~~ distribution ~~~~~~~~~'
    for apg in counter:
        print ' %s: %d' % (apg, counter[apg])
    print counter.keys()
    print '~~~~~~ lstm apgs ~~~~~ '
    for fu in lstm_apg:
        print 'FU: ', fu, '  APG: ', lstm_apg[fu]


def filter_ts_sales(df_ts_sales):
    if df_ts_sales is None:
        return None
    if df_ts_sales.shape[0] < 100:
        return None
    else:
        return df_ts_sales


def batch_run(ic_orders_by_year_pkl, ic_sales_by_year_pkl, results_csv=ssarima_results_csv):
    orders_by_year = pickle.load(open(ic_orders_by_year_pkl, 'rb'))
    sales_by_year = pickle.load(open(ic_sales_by_year_pkl, 'rb'))

    df_all_results = pd.DataFrame()
    t0 = time()
    ii = 0

    # for apg in orders_by_year:
    for apg in sales_by_year:
        for fu in sales_by_year[apg]:
            ii = ii + 1
            print '{ %d } <<<<<<< APG: %s\tFU: %s >>>>>>>' % (ii, apg, fu)
            ts_orders = select_timeseries(apg, fu, orders_by_year)
            ts_sales = select_timeseries(apg, fu, sales_by_year)

            df_ts_orders = make_time_series_df(ts_orders, 'orders')
            df_ts_sales = make_time_series_df(ts_sales, 'sales')
            df_ts_sales = filter_ts_sales(df_ts_sales)

            df_predict = rolling_ssarima_predictor(apg, fu, df_ts_orders, df_ts_sales)

            if df_predict is not None:
                df_all_results = df_all_results.append(df_predict)

    delta = time() - t0
    df_all_results.to_csv(results_csv)

    print '.... time taken: %d : %d' % (delta / 60, delta % 60)


def batch_control_run(ic_orders_by_year_pkl, ic_sales_by_year_pkl, results_csv='jcl_ssarima_results'):

    apg_list = ['UGB001', 'UGB002', 'UGB003', 'UGB004', 'UGB005', 'UGB006',
                'UGB008', 'UGB009', 'UGB012', 'UGB013', 'UGB014', 'UGB015', 'UGB016']

    orders_by_year = pickle.load(open(ic_orders_by_year_pkl, 'rb'))
    sales_by_year = pickle.load(open(ic_sales_by_year_pkl, 'rb'))

    df_all_results = pd.DataFrame()
    t0 = time()
    ii = 0
    apg = 'UGB016'
    for fu in orders_by_year[apg]:
        ii = ii + 1
        print '{ %d } <<<<<<< APG: %s\tFU: %s >>>>>>>' % (ii, apg, fu)
        ts_orders = select_timeseries(apg, fu, orders_by_year)
        ts_sales = select_timeseries(apg, fu, sales_by_year)

        df_ts_orders = make_time_series_df(ts_orders, 'orders')
        df_ts_sales = make_time_series_df(ts_sales, 'sales')
        df_ts_sales = filter_ts_sales(df_ts_sales)

        df_predict = rolling_ssarima_predictor(apg, fu, df_ts_orders, df_ts_sales)

        if df_predict is not None:
            df_all_results = df_all_results.append(df_predict)

    delta = time() - t0
    df_all_results.to_csv(results_folder + results_csv + '_' + apg + '.csv')

    print '.... time taken: %d : %d' % (delta / 60, delta % 60)
    print df_all_results


def vectorize(ic_orders_by_year_pkl, ic_sales_by_year_pkl):
    orders_by_year = pickle.load(open(ic_orders_by_year_pkl, 'rb'))
    sales_by_year = pickle.load(open(ic_sales_by_year_pkl, 'rb'))
    # ninst = sum(len(v) for v in sales_by_year.itervalues())
    # print '# instances: ', ninst
    orders_count, orders_apg_set, orders_fu_set, orders_freq = get_frequencies(orders_by_year)
    sales_count, sales_apg_set, sales_fu_set, sales_freq = get_frequencies(sales_by_year)

    # print '# confirmation: ', orders_count
    # print 'APG set: ', orders_apg_set
    # for apg in orders_fu_set:
    #     fu_list = sorted(list(orders_fu_set[apg]))
    #     # print apg
    #     # print '\t# in FU set: ', len(orders_fu_set[apg])
    #     # print '\tFU set: ', orders_fu_set[apg]
    #     # print '\tFU list: ', fu_list

    #  select randomly 5% of Orders FUs'
    random_orders_fu_list = OrderedDict()
    for apg in orders_fu_set:
        fu_list = sorted(list(orders_fu_set[apg]))
        rnelt = len(fu_list) / 20
        rlist = np.random.choice(fu_list, rnelt, replace=False)
        random_orders_fu_list[apg] = sorted(rlist)

    # print '\n------ randomly selected orders FU ------'
    # for apg in random_orders_fu_list:
    #     print apg, '(', len(random_orders_fu_list[apg]), '}: ', random_orders_fu_list[apg]

    # print '~~~~~~~~ Time eries Length Frequencies ~~~~~~~~'
    # for apg in orders_freq:
    #     print apg, ': ', orders_freq[apg]

    #  select randomly 10% of sales FUs'
    random_sales_fu_list = OrderedDict()
    for apg in sales_fu_set:
        fu_list = sorted(list(sales_fu_set[apg]))
        rnelt = len(fu_list) / 5
        rlist = np.random.choice(fu_list, rnelt, replace=False)
        random_sales_fu_list[apg] = sorted(rlist)

    # print '\n------ randomly selected Sales FU ------'
    # for apg in random_sales_fu_list:
    #     print apg, '(', len(random_sales_fu_list[apg]), '}: ', random_sales_fu_list[apg]

    # print '~~~~~~~~ Time eries Length Frequencies ~~~~~~~~'
    # for apg in orders_freq:
    #     print apg, ': ', orders_freq[apg]

    # apg = 'UGB001'
    # fu = 'IGB0007'

    # print '\n------  Sales Timeseries length FU ------'
    # for apg in random_sales_fu_list:
    #     for fu in random_sales_fu_list[apg]:
    #         ts_sales = select_timeseries(apg, fu, sales_by_year)
    #         df_ts_sales = make_time_series_df(ts_sales, 'sales')
    #         print apg, ' - ', fu, ': ', df_ts_sales.shape

    apg = 'UGB001'
    fu = 'IGB4804'

    apg = 'UGB001'
    fu = 'IGB0527'

    apg = 'UGB002'
    fu = 'IGB4840'

    apg = 'UGB004'
    fu = 'IGB4732'

    # fu = 'IGB4842'

    ts_orders = select_timeseries(apg, fu, orders_by_year)
    ts_sales = select_timeseries(apg, fu, sales_by_year)

    print '\n ~~~~~~~ Timeseries (%s, %s) ~~~~~~~ ' % (apg, fu)
    nelt = 0
    for year in orders_by_year[apg][fu]:
        for weeknum in orders_by_year[apg][fu][year]:
            # print weeknum, ': ', orders_by_year[apg][fu][year][weeknum]
            nelt = nelt + 1
    print ' # elt:  %d\n' % (nelt)

    df_ts_orders = make_time_series_df(ts_orders, 'orders')
    df_ts_sales = make_time_series_df(ts_sales, 'sales')

    # arima_predictor_ver1(apg, fu, df_ts_orders, df_ts_sales)

    # arima_predictor_ver1(apg, fu, df_ts_orders, df_ts_sales)

    svr_predictor(df_ts_orders)

    # lstm_predictor(apg, fu, df_ts_orders)

    # rolling_ssarima_predictor(apg, fu, df_ts_orders, df_ts_sales)

    # benchwork_predictions(orders_by_year, random_sales_fu_list)


def minimal_requirement_train(ic_orders_by_year_pkl):
    orders_by_year = pickle.load(open(ic_orders_by_year_pkl, 'rb'))
    apglist = ['UGB001', 'UGB002', 'UGB003', 'UGB004', 'UGB005', 'UGB006', 'UGB008',
               'UGB009', 'UGB012', 'UGB013', 'UGB014', 'UGB015']
    fu = 'IGB4842'
    rmse_list = []
    nrmse_list = []
    for apg in apglist:
        ts_orders = select_timeseries(apg, fu, orders_by_year)
        df_ts_orders = make_time_series_df(ts_orders, 'orders')
        rmse, normalized_rmse, y_hat, future = arima_predictor_ver1(apg, fu, df_ts_orders, None)
        rmse_list = rmse_list + [rmse]
        nrmse_list = nrmse_list + [normalized_rmse]
    x_rmse = np.array(rmse_list)
    x_nrmse = np.array(nrmse_list)
    print "=========== Prediction Quality for FU: %s and each APG in %s" % (fu, apglist)
    print 'avg RMSE: ', np.round(np.mean(x_rmse), decimals=2)
    print 'avg N-RMSE: ', np.round(np.mean(x_nrmse), decimals=4)


def make_fu_timeseries(fu_by_year):
    timeseries = OrderedDict()
    count_weeks = 0
    for fu in fu_by_year:
        if fu not in timeseries:
            timeseries[fu] = OrderedDict()
        for year in fu_by_year[fu]:
            for weeknum in fu_by_year[fu][year]:
                timeseries[fu][weeknum] = fu_by_year[fu][year][weeknum]
                count_weeks = count_weeks + 1
    return timeseries


def training_by_fu(fu_orders_by_year_pkl, fu_sales_by_year_pkl):
    fu_orders_by_year = pickle.load(open(fu_orders_by_year_pkl, 'rb'))
    fu_sales_by_year = pickle.load(open(fu_sales_by_year_pkl, 'rb'))
    orders_ts = make_fu_timeseries(fu_orders_by_year)
    sales_ts = make_fu_timeseries(fu_sales_by_year)
    print "# orders fus': ", len(orders_ts)
    print "# sales fus': ", len(sales_ts)
    zero_fu_orders = []
    for fu in orders_ts:
        do_print = fu in top_10_fu['IC']
        if do_print:
            print fu, ' (', len(orders_ts[fu]), ')'
        freq = 0
        sum = 0.0
        for weeknum in orders_ts[fu]:
            orders = orders_ts[fu][weeknum]
            if do_print:
                print '\t', weeknum, ': ', orders
            sum = sum + orders
            if orders > 0:
                freq = freq + 1

        if freq == 0:
            zero_fu_orders = zero_fu_orders + [fu]

    print '........ zero FU Orders .......'
    print zero_fu_orders

    fu = 'IGB4842'
    dateindex = orders_ts[fu].keys()
    df_ts_orders = pd.DataFrame.from_dict(orders_ts[fu], orient='index', columns=['orders'])
    # print df_ts_orders

    df_ts_sales = pd.DataFrame.from_dict(sales_ts[fu], orient='index', columns=['sales'])

    print '%s in sales? %s' % (fu, fu in sales_ts)

    rmse, normal_rmse, y_hat, future = arima_predictor_ver1('ALL', fu, df_ts_orders, None)
    print '\n============ Prediction Quality for FU: %s and all APGs ========' % (fu)
    print 'RSME = ', np.round(rmse, decimals=2)
    print 'Normalized RSME = ', np.round(normal_rmse, decimals=4)


if __name__ == '__main__':
    print 'Bayesian predictor ..'

    # dehyphen_weeknum(ic_orders_csv,ic_de_orders_csv)
    # to_orders_by_year(ic_de_orders_csv, ic_orders_by_year_pkl)
    # load_ds1_ic_orders(ds1_sub_ic_orders_pkl)

    # consolidate_to_FU(ic_orders_by_year_pkl, fu_orders_by_year_pkl)
    # consolidate_to_FU(ic_sales_by_year_pkl, fu_sales_by_year_pkl)

    # count_apg_fu_orders(ic_orders_by_year_pkl, ic_sales_by_year_pkl)

    # vectorize(ic_orders_by_year_pkl, ic_sales_by_year_pkl)

    rfr_predictor(ic_orders_by_year_pkl, ic_orders_by_week_csv)

    # batch_run(ic_orders_by_year_pkl, ic_sales_by_year_pkl, results_csv=ssarima_with_epos_results_csv)

    # batch_control_run(ic_orders_by_year_pkl, ic_sales_by_year_pkl)

    # lstm_batch_control_run(ic_orders_by_year_pkl, ic_sales_by_year_pkl)

    # minimal_requirement_train(ic_orders_by_year_pkl)

    # training_by_fu(fu_orders_by_year_pkl, fu_sales_by_year_pkl)

    # make_FU_APG_distributions(ic_orders_by_year_pkl, fu_distributions_apg_csv)

    # kantar_decompo(kantar_kmdt_csv)
    # kantar_pca_decompo(kantar_kmdt_csv)
    # kantar_select_features(kantar_kmdt_csv)
