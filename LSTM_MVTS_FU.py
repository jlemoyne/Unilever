'''
    (c) Anaplan 2018 - This program is the property of Anaplan
    initial code by Jean Claude L 2018-08-22

    versions    Date            by       Description
    ======================================================================
    0.2         2018-08-23      jcl     Transformation mechanism
    0.3         2018-08-24      jcl     Major runtime improvenment by
                                        clearing computing graph at each
                                        keras.background.session.clear()
    0.4         2018-08-25      jcl     Introduced cache to cacke results
                                        when tuning up

    To use this code please change the train_folder and results_folder
    you will also need ds2_orders_no_epos_sorted.csv which is in BigQuery
    under uk_trend under the name DS2A_orders_no_epos_sorted

    Nilesh:
            ... place your own remarks
    Sudipta:
            ... place your own remarks

'''

import sys
import os
import glob
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from collections import OrderedDict
import pickle
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
# from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras import backend as K
import numpy as np
from time import time
from datetime import datetime
from datetime import timedelta
import random

train_folder = '/Users/jeanclaudelemoyne/work/Data/Unilever/train/20180808/'
lstm_all_results_csv = train_folder + 'jcl_20180828_lstm_v12_fu_all_results.csv'
cache_folder = train_folder + 'cache/'

fu_orders_by_year_pkl = train_folder + 'fu_orders_by_year.pkl'
fu_sales_by_year_pkl = train_folder + 'fu_sales_by_year.pkl'

fu_vectorders_pkl = train_folder + 'fu_vectordes.pkl'
fu_sales_vectorders_pkl = train_folder + 'fu_sales_vectordes.pkl'

ic_apg_fu_sales_csv = train_folder + 'ic_apg_fu_sales.csv'
ic_fu_sales_pkl = train_folder + 'ic_fu_sales.pkl'

fu_orders_by_year = None
fu_sales_by_year = None
orders_fu = None
sales_fu = None

dataset = None
sales_dataset = None

num_ts = None


def consolidate(ic_apg_fu_sales_csv, ic_fu_sales_pkl):
    df_sales = read_csv(ic_apg_fu_sales_csv)
    fu_sales = OrderedDict()
    for index, row in df_sales.iterrows():
        fu = row['FU']
        weeknum = row['weeknum']
        sales = row['sales']
        if fu not in fu_sales:
            fu_sales[fu] = OrderedDict()
        if weeknum not in fu_sales[fu]:
            fu_sales[fu][weeknum] = 0
        fu_sales[fu][weeknum] += sales

    pickle.dump(fu_sales, open(ic_fu_sales_pkl, 'wb'))


def test_fu_sales():
    n = 0
    fu_sales_by_year = pickle.load(open(fu_sales_by_year_pkl, 'rb'))
    for fu in fu_sales_by_year:
        n += 1
        if n > 2:
            break
        for year in fu_sales_by_year[fu]:
            for weeknum in fu_sales_by_year[fu][year]:
                print weeknum, '\t', fu_sales_by_year[fu][year][weeknum]


def vectorize(fu_by_year, fu_vectors_pkl, fu_cols):

    if os.path.exists(fu_vectors_pkl):
        df_fus = pickle.load(open(fu_vectors_pkl, 'rb'))
        print df_fus.shape
        return df_fus

    # get period index first
    week_set = set()

    for fu in fu_by_year:
        for year in fu_by_year[fu]:
            for weeknum in fu_by_year[fu][year]:
                week_set.add(weeknum)
    week_index = sorted(list(week_set))

    df_fus = DataFrame(columns=fu_cols, index=week_index)
    print '------------ df_fus shape: ', df_fus.shape

    nn = len(fu_cols)
    n = 0
    for fu in fu_by_year:
        n += 1
        print '%s  .. %d / %d' % (fu, n, nn)
        for year in fu_by_year[fu]:
            for weeknum in fu_by_year[fu][year]:
                df_fus.at[weeknum, fu] = fu_by_year[fu][year][weeknum]
    print ' >>>>>>>>>>>> ', fu_vectors_pkl
    pickle.dump(df_fus, open(fu_vectors_pkl, 'wb'))

    return df_fus


def prepare_sales_vectors():
    global dataset, sales_dataset, num_ts
    # print 'dataset shape: ', dataset.shape
    # print 'sales_dataset shape: ', sales_dataset.shape
    dataset = dataset.drop([201553], axis=0)
    num_ts = dataset.shape[1]
    index1 = set(dataset.index)
    # sales_dataset = sales_dataset.sort_index()
    index2 = set(sales_dataset.index)
    c1 = index1 - index2
    c2 = index2 - index1
    tobe_removed = list(c2)

    sales_dataset = sales_dataset.drop(tobe_removed, axis=0)
    # rename columns in sales
    cols = sales_dataset.columns.values
    cols = ['x_' + col for col in cols]
    sales_dataset.columns = cols
    # concat
    dataset = pd.concat([dataset, sales_dataset], axis=1)


def initialize_data():
    global fu_orders_by_year, fu_sales_by_year, orders_fu, sales_fu, \
        dataset, sales_dataset

    fu_orders_by_year = pickle.load(open(fu_orders_by_year_pkl, 'rb'))
    orders_fu = sorted(fu_orders_by_year.keys())
    dataset = vectorize(fu_orders_by_year, fu_vectorders_pkl, orders_fu)
    # ignore SR_IC_NR
    dataset = dataset.drop(['SR_IC_NR'], axis=1)

    fu_sales_by_year = pickle.load(open(fu_sales_by_year_pkl, 'rb'))
    sales_fu = sorted(fu_sales_by_year.keys())
    sales_dataset = vectorize(fu_sales_by_year, fu_sales_vectorders_pkl, sales_fu)
    # print '~~~~~~~ index ~~~~~~~'
    # for weeknum in sales_dataset.index:
    #     print weeknum

    prepare_sales_vectors()

    # print sales_dataset.shape
    # print sales_dataset.head()


def feat_transform(fu, dataset):

    df_feat = dataset.drop([fu], axis=1)
    X = df_feat.values

    ts = dataset[fu]
    # print '~~~~~~ initial ~~~~~~~'
    # print ts
    # yrsum = OrderedDict()
    # print '----------------------'
    # for weeknum in ts.index:
    #     year = int(str(weeknum)[:4])
    #     if year not in yrsum:
    #         yrsum[year] = 0
    #     yrsum[year] += ts[weeknum]
    #     print ts[weeknum]

    # print '_____ yearly sum ------'
    # for year in yrsum:
    #     print '%d\t%d' % (year, yrsum[year])

    factors = ts.values

    # print ' ~~~~~~ factors ~~~~~~'
    # print factors.shape
    # for i in range(factors.shape[0]):
    #     print factors[i]
    #     if factors[i] > 1.0:
    #         factors[i] = 1
    #     else:
    #         factors[i] = 0
    df_fact = ts.ewm(com=0.25).mean()
    # print ' ~~~~~~ hot ~~~~~~'
    # print df_fact

    ema_fact = df_fact.values

    sum = np.sum(factors)

    # print 'sum = ', sum
    # A = np.array(ts.index).reshape(1, -1)
    # A = A.T
    # print ' y shape: ', y.shape
    # print 'A shape:', A.shape
    # beta_hat, residuals, rank, sv = np.linalg.lstsq(A, y, rcond=None)
    # print 'beta_hat: ', beta_hat

    # X_transfmd = np.log(X + 10 * np.ones(X.shape))
    # X_transfmd = np.random.rand(X.shape[0], X.shape[1])
    # X_transfmd = np.random.randn(X.shape[0], X.shape[1])

    # X_transfmd = np.random.poisson(100, size=X.shape)

    # df_feat_transfmd = DataFrame(X_transfmd, columns=df_feat.columns.values, index=df_feat.index)

    # X_transfmd = np.random.poisson(100, size=factors.shape)

    ema_fact = ema_fact.reshape(-1, 1)
    X_transfmd = np.concatenate((X, ema_fact), axis=1)
    cols = list(df_feat.columns.values) + ['Xformed']
    df_feat_transfmd = DataFrame(X_transfmd, columns=cols, index=df_feat.index)

    # print '______ df_feat_transformed ------'
    # print df_feat_transfmd
    #
    return df_feat_transfmd

    # return df_feat


# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('X%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('X%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('X%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)

    return agg


def make_trainset(train, test, i):
    train_X, train_y = np.delete(train, i, axis=1), train[:, i]
    test_X, test_y = np.delete(test, i, axis=1), test[:, i]

    return train_X, train_y, test_X, test_y


def fit_one_timeseries(dataset, ts_index, future=[(201810, 201814), (201814, 201819)], neurons=4, plot_graph=False):
    debug = True
    fu = dataset.columns.values[ts_index]
    index = []
    for from_index, to_index in future:
        index = index + range(from_index + 1, to_index + 1)

    df_predict = DataFrame(columns=['FU', 'predicted', 'actual', 'RMSE', 'N-RMSE', 'proba'], index=index)

    nrun = 0

    for from_index, to_index in future:
        nrun += 1
        # bring target to first column
        df_orders = dataset[fu]
        df_feat = feat_transform(fu, dataset)

        dataset = pd.concat([df_orders, df_feat], axis=1)

        values = dataset.values
        # ensure all data is float
        values = values.astype('float32')
        values = np.nan_to_num(values)

        # normalize features
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(values)
        # frame as supervised learning
        reframed = series_to_supervised(scaled, 1, 1)

        # print 'reframed shape: ', reframed.shape

        drop_cols = range(dataset.shape[1] + 1, reframed.shape[1])

        # drop columns we don't want to predict
        reframed.drop(reframed.columns[drop_cols], axis=1, inplace=True)
        # print(reframed.head())
        # print reframed.shape

        # split into train and test sets
        values = reframed.values
        offset = list(dataset.index).index(from_index)
        horizon = to_index - from_index

        # print '******* >>>>>>>> horizon: ', horizon

        n_train_weeks = offset
        train = values[:n_train_weeks, :]
        test = values[n_train_weeks:n_train_weeks + horizon, :]

        # print '******* >>>>>>> test shape: ', test.shape

        # print 'train shape: ', train.shape
        # print 'test shape: ', test.shape

        # train_X, train_y, test_X, test_y = make_trainset(train, test, ts_index)

        # split into input and outputs
        train_X, train_y = train[:, :-1], train[:, -1]
        test_X, test_y = test[:, :-1], test[:, -1]

        # print 'train_X shape: ', train_X.shape
        # print 'train_y shape: ', train_y.shape
        # print 'test_X shape: ', test_X.shape
        # print 'test_X shape: ', test_X.shape

        # reshape input to be 3D [samples, timesteps, features]
        train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
        test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
        # print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

        # design network
        model = Sequential()

        model.add(LSTM(neurons, input_shape=(train_X.shape[1], train_X.shape[2])))
        model.add(Dense(1))
        model.add(Activation('relu'))
        model.add(Dense(5, kernel_initializer='normal', activation='relu'))
        model.add(Dense(5, kernel_initializer='normal', activation='relu'))
        model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
        # model.add(Dropout(0.5))
        # model.add(Dense(1, activation='softmax'))
        # model.compile(loss='mean_squared_error', optimizer='sgd')
        # model.compile(loss='poisson', optimizer='sgd')
        model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
        # model.compile(loss='mae', optimizer='adam')
        # fit network
        history = model.fit(train_X, train_y, epochs=100, batch_size=500, validation_data=(test_X, test_y), verbose=0, shuffle=False)
        # plot history
        if plot_graph:
            pyplot.plot(history.history['loss'], label='train')
            pyplot.plot(history.history['val_loss'], label='test')
            pyplot.legend()
            pyplot.show()

        # make a prediction
        yhat = model.predict(test_X)
        yhat_proba = model.predict_proba(test_X)
        test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
        # invert scaling for forecast
        inv_yhat = concatenate((yhat, test_X[:, 1:]), axis=1)
        inv_yhat = scaler.inverse_transform(inv_yhat)
        inv_yhat = inv_yhat[:,0]
        # invert scaling for actual
        test_y = test_y.reshape((len(test_y), 1))
        inv_y = concatenate((test_y, test_X[:, 1:]), axis=1)
        inv_y = scaler.inverse_transform(inv_y)
        inv_y = inv_y[:,0]

        print 'ts index: %d ==== Prediction Quality FU: %s === ' % (ts_index, fu)
        # calculate RMSE
        stdev = np.std(inv_y)
        rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
        nrmse = rmse / stdev
        rmse = np.round(rmse, decimals=0)
        nrmse = np.round(nrmse, decimals=3)
        print('RMSE: %.0f\tN-RMSE: %.3f' % (rmse, nrmse))

        last_index = dataset.tail(1).index[0]

        index = range(from_index + 1, to_index + 1)

        week = from_index
        for x in np.nditer(inv_yhat):
            week = week + 1
            if week in index:
                df_predict.loc[week]['FU'] = fu
                df_predict.loc[week]['predicted'] = max(0, np.round(x, decimals=0))
                df_predict.loc[week]['RMSE'] = rmse
                df_predict.loc[week]['N-RMSE'] = nrmse

        week = from_index
        for x in np.nditer(inv_y):
            week = week + 1
            if week in index:
                df_predict.loc[week]['actual'] = x

        week = from_index
        for x in np.nditer(yhat_proba):
            week = week + 1
            if week in index:
                df_predict.loc[week]['proba'] = np.round(x, decimals=4)

    return fu, rmse, nrmse, df_predict


'''
        Run ALL APG x FUs'
'''


def run_all():

    initialize_data()

    t0 = time()
    df_all_predictions = DataFrame()
    nts = num_ts
    # print '****** ', nts
    cols = dataset.columns.values
    n = 0
    for ts_index in range(nts):
        tt0 = time()
        fu = cols[ts_index]
        K.clear_session()
        fu, rmse, nrmse, df_predict = fit_one_timeseries(dataset, ts_index,
                                                         future=[(201810, 201814), (201814, 201819)],
                                                         neurons=12,
                                                         plot_graph=False)
        n += 1
        print 'ts index: %d / %d ~~~~~~~~~====== df_redict (FU: %s) ======~~~~~~~~~' % (ts_index, nts, fu)
        print df_predict
        print '--------------------------'
        df_all_predictions = df_all_predictions.append(df_predict)
        ts_index = ts_index + 1

        ttdelta = time() - tt0
        ttcumul = time() - t0

        print '\n... This time series model fit took : %dmn : %ds  cumulative: %dmn : %ds' % \
              (ttdelta / 60, ttdelta % 60, ttcumul / 60, ttcumul % 60)

        # if ts_index > 3:
        #     break

    delta = time() - t0

    print '\n... computing time: %dmn : %ds' % (delta / 60, delta % 60)
    print '... %d/%d timeseries processed' % (n, nts)
    df_all_predictions.to_csv(lstm_all_results_csv)


if __name__ == '__main__':
    # consolidate(ic_apg_fu_sales_csv, ic_fu_sales_pkl)
    # test_fu_sales()
    # initialize_data()
    # feat_transform('IGB0007', dataset)
    run_all()
