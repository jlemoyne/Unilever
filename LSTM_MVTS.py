from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import numpy as np
from time import time

train_folder = '/Users/jeanclaudelemoyne/work/Data/Unilever/train/20180808/'
ds2_train_orders_csv = train_folder + 'ds2_train_orders.csv'
lstm_all_results_csv = train_folder + 'jcl_20180821_lstm_all_results.csv'


# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
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


def fit_one_timeseries(dataset, ts_index, from_index=201811, neurons=4, plot_graph=False):
    debug = True
    # load dataset
    apg_fu = dataset.columns.values[ts_index]
    apg, fu = apg_fu.split('-')
    # print '>>>>>>>>>> ', apg_fu, (apg, fu)
    # print 'dataset shape: ', dataset.shape

    values = dataset.values
    # ensure all data is float
    values = values.astype('float32')
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    # frame as supervised learning
    reframed = series_to_supervised(scaled, 1, 1)

    print 'reframed shape: ', reframed.shape

    drop_cols = range(dataset.shape[1] + 1, reframed.shape[1])

    # drop columns we don't want to predict
    reframed.drop(reframed.columns[drop_cols], axis=1, inplace=True)
    # print(reframed.head())
    # print reframed.shape

    # split into train and test sets
    values = reframed.values
    offset = list(dataset.index).index(from_index)

    n_train_weeks = offset
    train = values[:n_train_weeks, :]
    test = values[n_train_weeks:, :]

    # print 'train shape: ', train.shape
    # print 'test shape: ', test.shape

    train_X, train_y, test_X, test_y = make_trainset(train, test, ts_index)

    # split into input and outputs
    # train_X, train_y = train[:, :-1], train[:, -1]
    # test_X, test_y = test[:, :-1], test[:, -1]

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
    history = model.fit(train_X, train_y, epochs=100, batch_size=500, validation_data=(test_X, test_y), verbose=2, shuffle=False)
    # plot history
    if plot_graph:
        pyplot.plot(history.history['loss'], label='train')
        pyplot.plot(history.history['val_loss'], label='test')
        pyplot.legend()
        pyplot.show()

    # make a prediction
    yhat = model.predict(test_X)
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

    # print '~~~~~~~ inv_yhat ~~~~~~~ ', inv_yhat.shape, ' ::: ', type(inv_yhat)
    # print inv_yhat

    print 'ts index: %d ==== Prediction Quality APG: %s   FU: %s === ' % (ts_index, apg, fu)
    # calculate RMSE
    stdev = np.std(inv_y)
    rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
    nrmse = rmse / stdev
    rmse = np.round(rmse, decimals=0)
    nrmse = np.round(nrmse, decimals=3)
    print('RMSE: %.0f\tN-RMSE: %.3f' % (rmse, nrmse))

    last_index = dataset.tail(1).index[0]
    index = range(from_index + 1, last_index + 1)
    df_predict = DataFrame(columns=['APG', 'FU', 'predicted', 'actual', 'RMSE', 'N-RMSE'], index=index)

    week = from_index
    for x in np.nditer(inv_yhat):
        week = week + 1
        df_predict.loc[week]['APG'] = apg
        df_predict.loc[week]['FU'] = fu
        df_predict.loc[week]['predicted'] = max(0, x)
        df_predict.loc[week]['RMSE'] = rmse
        df_predict.loc[week]['N-RMSE'] = nrmse

    week = from_index
    for x in np.nditer(inv_y):
        week = week + 1
        df_predict.loc[week]['actual'] = x

    return apg, fu, df_predict


if __name__ == '__main__':
    dataset = read_csv(ds2_train_orders_csv, header=0, index_col=0)
    nts = dataset.shape[1]
    t0 = time()
    df_all_predictions = DataFrame()
    n = 0
    for ts_index in range(400, nts):
        apg, fu, df_predict = fit_one_timeseries(dataset, ts_index, plot_graph=False)
        n = n + 1
        print 'ts index: %d ~~~~~~~~~====== df_redict (APG: %s, FU: %s) ======~~~~~~~~~' % (ts_index, apg, fu)
        print df_predict
        df_all_predictions = df_all_predictions.append(df_predict)
        exit(101)

    delta = time() - t0
    print '\n... computing time: %02d:%02d' % (delta / 60, delta % 60)
    print '... %d/%d timeseries processed' % (n, nts)
    df_all_predictions.to_csv(lstm_all_results_csv)
