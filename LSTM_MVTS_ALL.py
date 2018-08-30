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
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
# from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras import backend as K
import numpy as np
from time import time
import random

train_folder = '/Users/jeanclaudelemoyne/work/Data/Unilever/train/20180808/'
lstm_all_results_csv = train_folder + 'jcl_20180825_lstm3_all_results.csv'
cache_folder = train_folder + 'cache/'

optimum_folder = train_folder + 'optimum/'
optimum_features_txt = optimum_folder + 'best_features.txt'
optimum_features_pkl = optimum_folder + 'best_features.pkl'
optimum_ranked_features_txt = optimum_folder + 'best_ranked_features.txt'

ds2_orders_no_epos_csv = train_folder + 'ds2_orders_no_epos_sorted.csv'
dump1ts_csv = train_folder + 'dump1ts.csv'
apgxfu_pkl = train_folder + 'APGxFU.pkl'

poisson_dist_pkl = train_folder + 'poisson_features.pkl'
transfeat_pkl = 'transfeat.pkl'
random_features_pkl = train_folder + 'hold/poisson_features.pkl'

# dataset training source all features
df_a = None
apg_fu = None
rand_features = None


def prepare_data():
    global df_a
    df_a = df_a.set_index('Week_Year')
    df_a = df_a.drop(['APG_desc', 'FU_desc', 'Flavors', 'ana_category',
                      'Calendar_Week_Number',
                      'UGB001_promo_key_before', 'UGB002_promo_key_before',
                      'UGB003_promo_key_before', 'UGB004_promo_key_before',
                      'UGB005_promo_key_before', 'UGB006_promo_key_before',
                      'UGB008_promo_key_before', 'UGB009_promo_key_before',
                      'UGB012_promo_key_before', 'UGB013_promo_key_before',
                      'UGB014_promo_key_before', 'UGB015_promo_key_before',
                      'UGB016_promo_key_before',
                      'X_week_before_promo', 'X_week_after_promo',
                      'UGB001_promo_key_after', 'UGB002_promo_key_after',
                      'UGB003_promo_key_after', 'UGB004_promo_key_after',
                      'UGB005_promo_key_after', 'UGB006_promo_key_after',
                      'UGB008_promo_key_after', 'UGB009_promo_key_after',
                      'UGB012_promo_key_after', 'UGB013_promo_key_after',
                      'UGB014_promo_key_after', 'UGB015_promo_key_after',
                      'UGB016_promo_key_after', 'same_week_promo'],
                     axis=1)
    # print df_a.columns.values


def initialize_data():
    global df_a, apg_fu
    if df_a is None:
        df_a = read_csv(ds2_orders_no_epos_csv)
        prepare_data()
        print df_a.head()

    if os.path.exists(apgxfu_pkl):
        apgxfu = pickle.load(open(apgxfu_pkl, 'rd'))
    else:
        apgxfu = OrderedDict()
        n = 0
        for index, row in df_a.iterrows():
            apg = row['APG']
            fu = row['FU']
            if apg not in apgxfu:
                apgxfu[apg] = OrderedDict()
            apgxfu[apg][fu] = None
            n = n + 1

        pickle.dump(apgxfu, open(apgxfu_pkl, 'wb'))
        print ' # APG x FU read: ', n
        n = 0
        for apg in apgxfu:
            n = n + len(apgxfu[apg])
        print ' # Unique APG x FU: ', n

    for apg in apgxfu:
        for fu in apgxfu[apg]:
            if apg_fu is None:
                apg_fu = []
            apg_fu = apg_fu + [(apg, fu)]
    print '# Timeseries: ', len(apg_fu)


def make_timeseries(apg, fu, drop_ratio=0.6):
    global df_a

    df_series = df_a.query('APG == "%s" and FU == "%s"' % (apg, fu))
    df_series = df_series.drop(['APG', 'FU'], axis=1)

    df_orders = df_series['Qty_ordered']
    df_series = df_series.drop(['Qty_ordered'], axis=1)

    ncols = len(df_series.columns.values)
    rand_colx = np.random.randint(0, high=ncols, size= int(ncols * drop_ratio))
    df_series = df_series.drop(df_series.columns[rand_colx], axis=1)

    df_series.insert(0, 'orders', df_orders)

    return df_series


def feat_transform(apg, fu, df_series, override=False, ratio=3.5):
    global rand_features

    y = df_series['orders'].values
    X = df_series.drop(['orders'], axis=1).values

    # X_transfmd = np.log(X + 10 * np.ones(X.shape))
    # X_transfmd = np.random.rand(X.shape[0], X.shape[1])
    # X_transfmd = np.random.randn(X.shape[0], X.shape[1])

    if rand_features is None or override:
        X_transfmd = np.random.poisson(100, size=X.shape)
        pickle.dump(X_transfmd, open(poisson_dist_pkl, 'wb'))
        if override:
            cache_name = cache_folder + '%s_%s_%.3f_poisson_%s' % (apg, fu, ratio, transfeat_pkl)
            pickle.dump(X_transfmd, open(cache_name, 'wb'))

    else:
        X_transfmd = rand_features

    # used to get the rank of the matrix
    beta_hat, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
    y = y.reshape(y.shape[0], -1)
    revalues = np.concatenate((y, X_transfmd), axis=1)
    df_series_transfmd = DataFrame(revalues, columns=df_series.columns.values, index=df_series.index)

    return df_series_transfmd

    # return df_series


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


def fit_one_timeseries(apg, fu, mv_time_series, ts_index, from_index=201810, to_index=201819,
                       neurons=4, plot_graph=False):
    values = mv_time_series.values
    # ensure all data is float
    values = values.astype('float32')
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    # frame as supervised learning
    reframed = series_to_supervised(scaled, 1, 1)

    drop_cols = range(mv_time_series.shape[1] + 1, reframed.shape[1])

    # drop columns we don't want to predict
    reframed.drop(reframed.columns[drop_cols], axis=1, inplace=True)

    # print(reframed.head())
    # print reframed.shape

    # split into train and test sets
    values = reframed.values
    offset = list(mv_time_series.index).index(from_index)

    n_train_weeks = offset
    train = values[:n_train_weeks, :]
    test = values[n_train_weeks:, :]

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
    model.compile(loss='mae', optimizer='adam')
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

    print 'ts index: %d ==== Prediction Quality APG: %s   FU: %s === ' % (ts_index, apg, fu)
    # calculate RMSE
    stdev = np.std(inv_y)
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    nrmse = np.NAN
    if stdev > 0.0:
        nrmse = rmse / stdev
    rmse = np.round(rmse, decimals=0)
    nrmse = np.round(nrmse, decimals=3)
    print('RMSE: %.0f\tN-RMSE: %.3f' % (rmse, nrmse))

    last_index = mv_time_series.tail(1).index[0]
    index = range(from_index + 1, to_index + 1)
    df_predict = DataFrame(columns=['APG', 'FU', 'predicted', 'actual', 'RMSE', 'N-RMSE', 'proba'], index=index)

    week = from_index
    for x in np.nditer(inv_yhat):
        week = week + 1
        if week in index:
            df_predict.loc[week]['APG'] = apg
            df_predict.loc[week]['FU'] = fu
            df_predict.loc[week]['predicted'] = max(0, np.round(x, decimals=0))
            df_predict.loc[week]['RMSE'] = rmse
            df_predict.loc[week]['N-RMSE'] = nrmse

    week = from_index
    for x in np.nditer(inv_y):
        week = week + 1
        if week in index:
            df_predict.loc[week]['actual'] = np.round(x, decimals=0)

    week = from_index
    for x in np.nditer(yhat_proba):
        week = week + 1
        if week in index:
            df_predict.loc[week]['proba'] = np.round(x, decimals=4)

    return rmse, nrmse, df_predict


'''
        Run ALL APG x FUs'
'''


def load_random_features():
    if os.path.exists(random_features_pkl):
        rand_features = pickle.load(open(random_features_pkl, 'rb'))
        if rand_features is not None:
            print 'Random features loaded ... from file ', random_features_pkl
    else:
        print '*** ', random_features_pkl, ' not found!!!'


def run_all():
    initialize_data()
    load_random_features()
    nts = len(apg_fu)
    print '# nts: ', nts

    t0 = time()
    df_all_predictions = DataFrame()
    ts_index = 0
    n = 0
    for apg, fu in apg_fu:
        tt0 = time()
        K.clear_session()
        mv_time_series = make_timeseries(apg, fu)
        rmse, nrmse, df_predict = fit_one_timeseries(apg, fu, mv_time_series, ts_index, plot_graph=False)
        n += 1
        print 'ts index: %d ~~~~~~~~~====== df_redict (APG: %s, FU: %s) ======~~~~~~~~~' % (ts_index, apg, fu)
        print df_predict
        print '--------------------------'
        df_all_predictions = df_all_predictions.append(df_predict)
        ts_index = ts_index + 1
        ttdelta = time() - tt0
        ttcumul = time() - t0
        print '\n... This time series model fit took: %02d:%02d  cumulative: %02d:%02d' % \
              (ttdelta / 60, ttdelta % 60, ttcumul / 60, ttcumul % 60)
        # if ts_index > 0:
        #     exit(101)

    delta = time() - t0
    print '\n... computing time: %02d:%02d' % (delta / 60, delta % 60)
    print '... %d/%d timeseries processed' % (n, nts)
    df_all_predictions.to_csv(lstm_all_results_csv)


'''
    Note: The Graph plots Loss v/s Epoch
'''


def clear_cache():
    files = glob.glob(cache_folder + '*')
    for f in files:
        # os.remove(f)
        print ' cache not cleared ... interrupted!'
        exit(301)


def find_optimum():
    pre_cols = pickle.load(open(optimum_features_pkl, 'rb'))
    freq = OrderedDict()
    for col in pre_cols:
        if col != 'orders':
            freq[col] = 0
    files = glob.glob(cache_folder + '*')
    min_nrmse = sys.float_info.max
    min_apg_fu = None
    min_cols = None
    min_ratio = None
    max_proba = sys.float_info.min
    max_apg_fu = None
    max_cols = None
    nn = 0
    for f in files:
        fnamex = os.path.basename(f)
        if fnamex.find('_perform') > -1:
            nn += 1
            fname_list = fnamex.split('.')
            apg_fu = fname_list[0].split('_')
            drop_list = fname_list[1].split('_')
            apg = apg_fu[0]
            fu = apg_fu[1]
            ratio = drop_list[0]
            cols, rmse, nrmse, df_predict = pickle.load(open(f, 'rb'))

            for index, row in df_predict.iterrows():
                proba = row['proba']
                if proba > max_proba:
                    max_apg_fu = (apg, fu)
                    max_proba = proba

            print apg, '\t', fu, '\t', ratio, rmse, nrmse
            if nrmse < min_nrmse:
                min_nrmse = nrmse
                min_apg_fu = (apg, fu)
                min_cols = cols
                min_ratio = ratio

            for col in freq.keys():
                if col in cols:
                    freq[col] += 1

    weight = OrderedDict()
    for col in freq:
        weight[col] = float(freq[col]) / float(nn)

    g = open(optimum_ranked_features_txt, 'w')
    print '~~~~~~~ ranked by weight ~~~~~~~~'
    for key, value in sorted(weight.iteritems(), key=lambda (k,v): (v,k)):
        g.write(key + '\n')
        print "%s: %s" % (key, value)
    g.close()

    print '~~~~~~~ min RMSE ~~~~~~~~'
    print 'RMSE: ', min_nrmse
    print 'min APG, FU: ', min_apg_fu
    print 'ratio: ', min_ratio
    print 'max proba: ', proba
    print 'max APG, FU: ', max_apg_fu
    print len(min_cols), min_cols
    g = open(optimum_features_txt, 'w')
    for col in min_cols:
        g.write(col + '\n')
    g.close()
    pickle.dump(min_cols, open(optimum_features_pkl, 'wb'))


def tune_up():
    interactive = False
    initialize_data()
    load_random_features()
    max_nts = len(apg_fu)
    sample = random.sample(apg_fu, max_nts / 5)
    nsample = len(sample)
    print sample
    t0 = time()
    if not os.path.exists(cache_folder):
        os.mkdir(cache_folder)
    clear_cache()
    #  plot loss (error) over number of epoch
    # for drop_ratio in [0.2, 0.3, 0.35, 0.4]:
    for drop_ratio in [0.3, 0.35, 0.4, 0.45]:
        ts_index = 0
        for apg, fu in sample:
            tt0 = time()
            K.clear_session()
            mv_time_series = feat_transform(apg, fu, make_timeseries(apg, fu, drop_ratio=drop_ratio), override=True,
                                            ratio=drop_ratio)
            rmse, nrmse, df_predict = fit_one_timeseries(apg, fu, mv_time_series, ts_index, plot_graph=interactive)
            cache_name = cache_folder + '%s_%s_%.3f_perform.pkl' % (apg, fu, drop_ratio)
            pickle.dump((mv_time_series.columns.values, rmse, nrmse, df_predict), open(cache_name, 'wb'))
            ts_index += 1
            print 'ts index: %d/%d %.3f ~~~~~~~~~====== df_redict (APG: %s, FU: %s) ======~~~~~~~~~' % \
                  (ts_index, nsample, drop_ratio, apg, fu)
            print df_predict
            print '--------------------------'
            ttdelta = time() - tt0
            ttcumul = time() - t0
            print '\n... This time series model fit took: %02d:%02d  cumulative: %02d:%02d' % \
                  (ttdelta / 60, ttdelta % 60, ttcumul / 60, ttcumul % 60)

            if interactive:
                answer = raw_input('q(uit) or any? ')
                print '>>>>> answer: ', answer
                if len(answer) > 0:
                    answer = answer[0].lower()
                    if answer[0] == 'q':
                        exit(301)


if __name__ == '__main__':
    run_all()
    # tune_up()
    # find_optimum()
