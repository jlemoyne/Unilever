import os
import pickle
from collections import OrderedDict
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
import numpy as np
import math
import statsmodels.tsa
import statsmodels.api as sm

train_folder_20180803 = '/Users/jeanclaudelemoyne/work/Data/Unilever/train/20180803/'
train_dataset = OrderedDict()
train_dataset['ic_sorted_ds1'] = train_folder_20180803 + 'train_ic_sorted_ds1.csv'
train_dataset['ic_ds1'] = train_folder_20180803 + 'train_ic_ds1.csv'
train_dataset['ic_ds2'] = train_folder_20180803 + 'train_ic_ds2.csv'
train_dataset['tea_ds1'] = train_folder_20180803 + 'train_tea_ds1.csv'
train_dataset['tea_ds2'] = train_folder_20180803 + 'train_tea_ds2.csv'
trainset = OrderedDict()
trainset['ic_ds1_sub1'] = train_folder_20180803 + 'train_ic_ds1_sub1.pkl'
epos_data_csv = train_folder_20180803 + 'epos_by_apg_fu_weekly_sales.csv'
epos_sales_by_year_pkl = train_folder_20180803 + 'epos_sales_by_year.pkl'
epos_reguralized_sales_by_year_pkl = train_folder_20180803 + 'epos_reguralized_sales_by_year.pkl'
epos_reguralized_sales_by_year_csv = train_folder_20180803 + 'epos_reguralized_sales_by_year.csv'

trainset2 = OrderedDict()
train_folder_20180808 = '/Users/jeanclaudelemoyne/work/Data/Unilever/train/20180808/'
trainset2['ds1_ic_orders'] = train_folder_20180808 + 'DS1_With_EPOS_Count10_IC_Order.csv'
trainset2['ds1_ic_sales'] = train_folder_20180808 + 'DS1_With_EPOS_Count10_IC_Sale_sorted.csv'
trainset2['ds1_sub_ic_orders'] = train_folder_20180808 + 'ds1_sub_ic_orders.pkl'
trainset2['ds1_sub_ic_sales'] = train_folder_20180808 + 'ds1_sub_ic_sales.pkl'
trainset2['ic_sales_by_year'] = train_folder_20180808 + 'ic_sales_by_year.pkl'
trainset2['ic_regularized_sales_by_year'] = train_folder_20180808 + 'ic_regularized_sales_by_year.pkl'
trainset2['ic_regularized_sales_csv'] = train_folder_20180808 + 'ic_regularized_sales.csv'


def check_training():
    check = True
    for ds in train_dataset:
        check = check and os.path.exists(train_dataset[ds])
    print 'pass train datasets check: ', check
    return check


feature_selection = OrderedDict()
feature_selection['basic_sans_promo'] = ['APG_Code', 'FU', 'year_week', 'Qty_ordered',
                     'Tot_Sls_Unit_Vol_Qt',
                      'Price', 'Upc_Ean_Count',
                     'Less_12_Male', 'Between_12_17_Male', 'Between_18_24_Male',
                     'Between_25_34_Male', 'Between_35_44_Male', 'Between_45_54_Male',
                     'Between_55_64_Male', 'MoreThan_65_Male', 'Less_12_Female',
                     'Between_12_17_Female', 'Between_18_24_Female', 'Between_25_34_Female',
                     'Between_35_44_Female', 'Between_45_54_Female', 'Between_55_64_Female',
                     'MoreThan_65_Female', 'Mean_Income', 'Median_Income']


def load_train_dataset(dsname):
    df = pd.read_csv(train_dataset[dsname], low_memory=False)
    return df.columns.values, df


def load_dataset(train_csv):
    df = pd.read_csv(train_csv, low_memory=False)
    return df.columns.values, df


def subset_train_dataset(df_train, cols):
    df_subset = df_train[cols]
    return df_subset


def store_trainset(df, train_pkl):
    pickle.dump(df, open(train_pkl, 'wb'))


def load_trainset(train_pkl):
    return pickle.load(open(train_pkl, 'rb'))


def eg_MLP():
    x_train = np.random.random((1000, 20))
    y_train = keras.utils.to_categorical(np.random.randint(10, size=(1000, 1)), num_classes=10)
    x_test = np.random.random((100, 20))
    y_test = keras.utils.to_categorical(np.random.randint(10, size=(100, 1)), num_classes=10)

    model = Sequential()
    # Dense(64) is a fully-connected layer with 64 hidden units.
    # in the first layer, you must specify the expected input data shape:
    # here, 20-dimensional vectors.
    model.add(Dense(64, activation='relu', input_dim=20))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=20, batch_size=128)
    score = model.evaluate(x_test, y_test, batch_size=128)


def dump_yearly_sales_orders(yearly_sales, yearly_orders):
    g = open(train_folder_20180803 + 'ic_ds1_yearly_gaps.txt', 'w')
    print "======== yearly sales ======="
    g.write("======== yearly sales =======\n")
    for apg in yearly_sales:
        print apg
        g.write(apg + '\n')
        for fu in yearly_sales[apg]:
            print '\t', fu
            g.write('\t' + fu + '\n')
            for year in yearly_sales[apg][fu]:
                matched = ' *** Not Matched'
                if apg in yearly_orders:
                    if fu in yearly_orders[apg]:
                        if year in yearly_orders[apg][fu]:
                            matched = ' === MATCHED'

                g.write('\t\t%s\tsales = %9.0f\t%s\n' % (year, yearly_sales[apg][fu][year], matched))
                print '\t\t%s\tsales = %9.0f\t%s' % (year, yearly_sales[apg][fu][year], matched)

    print "======== yearly orders ======="
    g.write("======== yearly orders =======\n")
    for apg in yearly_orders:
        print apg
        g.write(apg + '\n')
        for fu in yearly_orders[apg]:
            print '\t', fu
            g.write('\t' + fu + '\n')
            for year in yearly_orders[apg][fu]:
                matched = ' *** Not Matched'
                if apg in yearly_sales:
                    if fu in yearly_sales[apg]:
                        if year in yearly_sales[apg][fu]:
                            matched = ' === MATCHED'
                g.write('\t\t%s\torders = %9.0f\t%s\n' % (year, yearly_orders[apg][fu][year], matched))
                print '\t\t%s\torders = %9.0f\t%s' % (year, yearly_orders[apg][fu][year], matched)

    g.close()


def randomize_missing(g, wknb, apg, fu, year, weekly_data):
    ndp = len(weekly_data)
    g.write('%s\t%s\t%s # weekly data points: %d\n' % (apg, fu, year, ndp))
    print '%s\t%s\t%s # weekly data points: %d' % (apg, fu, year, ndp)
    nndp = 0
    first_weeknum = None
    last_weeknum = None
    wts = OrderedDict()
    for wk in wknb:
        wts[wk] = None
    for weeknum in weekly_data:
        if first_weeknum is None:
            first_weeknum = weeknum
        last_weeknum = weeknum
        if not math.isnan(weekly_data[weeknum]):
            wts[weeknum] = weekly_data[weeknum]
            nndp = nndp + 1

    g.write('\tFirst week#: %s\tLast week#: %s\n' % (first_weeknum, last_weeknum))
    print '\tFirst week#: %s\tLast week#: %s' % (first_weeknum, last_weeknum)
    if nndp != ndp:
        g.write('\t\t%s\t%s\t%s ******* 000000 ******* total: %d ?? non-null: %d\n' % (apg, fu, year, ndp, nndp))
        print '\t\t%s\t%s\t%s ******* 000000 ******* total: %d ?? non-null: %d' % (apg, fu, year, ndp, nndp)

    return ndp, nndp, weekly_data


def check_data(df_trainset):
    nn = 0; kk = 0
    sales_by_year = OrderedDict()
    for index, row in df_trainset.iterrows():
        nn = nn + 1
        if nn % 1000 == 0:
            print ' ... ', nn
        apg = row['APG_Code']
        fu = row['FU']
        weeknum = row['year_week']
        population_12 = row['Less_12_Male'] + row['Less_12_Female']
        population_44 = row['Between_35_44_Male'] + row['Between_35_44_Female']
        population_54 = row['Between_45_54_Male'] + row['Between_45_54_Female']
        sales_qty = row['Tot_Sls_Unit_Vol_Qt']
        orders_qty = row['Qty_ordered']
        year = str(weeknum)[:4]

        if apg not in sales_by_year:
            sales_by_year[apg] = OrderedDict()
        if fu not in sales_by_year[apg]:
            sales_by_year[apg][fu] = OrderedDict()
        if year not in sales_by_year[apg][fu]:
            sales_by_year[apg][fu][year] = OrderedDict()
        sales_by_year[apg][fu][year][weeknum] = sales_qty

    g = open(train_folder_20180803 + 'ic_ds1_missing_data.txt', 'w')
    # missing data treatment
    x_ndp = []
    x_nndp = []

    for apg in sales_by_year:
        for fu in sales_by_year[apg]:
            for year in sales_by_year[apg][fu]:
                wknb = range(1, 53)
                wknb = [year + '%02d' % x for x in wknb]
                ndp, nndp, sales_by_year[apg][fu][year] = randomize_missing(g, wknb, apg, fu, year, sales_by_year[apg][fu][year])
                x_ndp = x_ndp + [ndp]
                x_nndp = x_nndp + [nndp]
    xndp = np.array(x_ndp)
    xnndp = np.array(x_nndp)
    print ' avg # data points: ', int(np.mean(xndp)), '  std dev: ', int(np.std(xndp))
    print ' avg # non-null data points: ', int(np.mean(xnndp)), '  std dev: ', int(np.std(xnndp))
    print ' max # data points: ', np.max(xndp)
    print 'max # non-null data points: ', np.max(xnndp)
    ndp_med = np.median(xndp)
    nndp_med = np.median(xnndp)
    print 'median # data points: ', ndp_med, ' % from 52 weeks: ', int(float(ndp) * 100 / 52.0)
    print 'median # non-null data points: ', nndp_med, ' % from 52 weeks: ',  int(float(nndp_med) * 100  / 52.0)
    # print wknb
    g.close()


def estimate_sales(X, Y, projected_week):
    x = np.array(X)
    y = np.array(Y)
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return int(m * int(projected_week) + c)


def regularize_epos_data(ts, prev_year_sales, week_periods):
    if len(ts) > 51:
        sum_sales = 0
        for weeknum in ts:
            sum_sales = sum_sales + ts[weeknum]
        print week_periods[0], ' ... satisfied! .. ', sum_sales
        return ts

    full_ts = OrderedDict()
    for period in week_periods:
        full_ts[period] = None
    sales_qty = []
    for weeknum in ts:
        sales_qty = sales_qty + [ts[weeknum]]
        full_ts[weeknum] = ts[weeknum]
    x = np.array(sales_qty)

    mean = np.mean(x)
    std = np.std(x)

    sum_sales = 0
    weeks = full_ts.keys()
    for index in range(len(weeks)):
        weeknum = weeks[index]
        if full_ts[weeknum] is None:
            X = []; Y = []
            for j in range(index):
                X = X + [ full_ts[weeks[j]] ]
                Y = Y + [weeks[j]]
            if len(X) > 2:
                full_ts[weeknum] = estimate_sales(X, Y, weeknum)
            else:
                full_ts[weeknum] = max(0, np.random.normal(mean, std))
        sum_sales = sum_sales + full_ts[weeknum]

    print week_periods[0], ' .... completed! ....', sum_sales

    return full_ts


def incude_epos_data(sales_by_year, ts_min_len):
    wknb = range(1, 53)
    nacc = 0
    nelt = sum(len(v) for v in sales_by_year.itervalues())
    print '# elements: ', nelt
    for apg in sales_by_year:
        print apg
        for fu in sales_by_year[apg]:
            # print '\t', fu
            prev_year = None
            for year in sales_by_year[apg][fu]:
                week_periods = [year + '%02d' % int(x) for x in wknb]
                ts_len = len(sales_by_year[apg][fu][year])
                prev_year_sales = None
                if prev_year is not None:
                    prev_year_sales = sales_by_year[apg][fu][prev_year]
                sales_by_year[apg][fu][year] = regularize_epos_data(sales_by_year[apg][fu][year], prev_year_sales,
                                                                    week_periods)
                ts_len = len(sales_by_year[apg][fu][year])
                if ts_len >= ts_min_len:
                    # print '\t\t', year, ' :: ', ts_len
                    # sales_by_year[apg][fu][year] = regularize_epos_data(sales_by_year[apg][fu][year], week_periods)
                    nacc = nacc + 1
                prev_year = year
    return nacc, sales_by_year


def extrapolate_epos_data(epos_sales_by_year_pkl, epos_reguralized_sales_by_year_pkl):
    sales_by_year = pickle.load(open(epos_sales_by_year_pkl, 'rb'))
    nacc, regularized_sales_by_year = incude_epos_data(sales_by_year, 40)
    pickle.dump(regularized_sales_by_year, open(epos_reguralized_sales_by_year_pkl, 'wb'))
    print ' # acceptable TS: ', nacc


def eops_regularized_tocsv(epos_reguralized_sales_by_year_pkl, epos_reguralized_sales_by_year_csv):
    reg_sales = pickle.load(open(epos_reguralized_sales_by_year_pkl, 'rb'))
    g = open(epos_reguralized_sales_by_year_csv, 'w')
    g.write('APG,FU,WEEKNUM,SALES\n')
    for apg in reg_sales:
        for fu in reg_sales[apg]:
            for year in reg_sales[apg][fu]:
                for weeknum in reg_sales[apg][fu][year]:
                    sales_qty = reg_sales[apg][fu][year][weeknum]
                    g.write('%s,%s,%s,%d\n' % (apg, fu, weeknum, sales_qty))
    g.close()


def analyze_epos_data(epos_data_csv, epos_sales_by_year_pkl):
    print '\nanalyze_epos_data ..'
    print 'reading ', epos_data_csv, '... '
    df = pd.read_csv(epos_data_csv)
    print df.shape
    sales_by_year = OrderedDict()
    for index, row in df.iterrows():
        apg = row['APG']
        fu = row['FU']
        week = row['weeknum']
        year = str(week)[:4]
        sales = row['sales']
        if apg not in sales_by_year:
            sales_by_year[apg] = OrderedDict()
        if fu not in sales_by_year[apg]:
            sales_by_year[apg][fu] = OrderedDict()
        if year not in sales_by_year[apg][fu]:
            sales_by_year[apg][fu][year] = OrderedDict()
        sales_by_year[apg][fu][year][week] = sales

    pickle.dump(sales_by_year, open(epos_sales_by_year_pkl, 'wb'))

    x = []
    nnul = 0
    nn = 0
    for apg in sales_by_year:
        print '%s # FU: %d' % (apg, len(sales_by_year[apg]))
        for fu in sales_by_year[apg]:
            print '\t%s # Year: %d' % (fu, len(sales_by_year[apg][fu]))
            for year in sales_by_year[apg][fu]:
                nweek = len(sales_by_year[apg][fu][year])
                x = x + [nweek]
                for week in sales_by_year[apg][fu][year]:
                    nn = nn + 1
                    if math.isnan(sales_by_year[apg][fu][year][week]):
                        nnul = nnul + 1
                print '\t\t%s # weeks: %d' % (year, nweek)

    nacc = 0
    for i in range(53, 40, -1):
        kk = x.count(i)
        nacc = nacc + kk
        print ' # ', i, ' : ', kk, ' / ', len(x)
    print ' # acceptable: ', nacc, ' / ', len(x)

    print ' # of acceptable ts data: ', incude_epos_data(sales_by_year, 40)

    xl = np.array(x)
    m = len(x) * 52
    print '~ median: ', np.median(xl)
    print '~ mean: ', int(np.mean(xl))
    print '~ std: ', int(np.std(xl))
    print '# null: ', nnul, ' / ', nn
    print '% weeks: ', float(nn) * 100.0 / float(m)


'''
    Dataset Transformation Operations
'''


def subset_ds1_ic_orders(ds1_ic_orders_csv, ds1_sub_ic_orders_pkl):
    # cols = ['APG','APG_desc','FU','FU_desc','ana_category','Week_Year','Qty_ordered']
    subcols = ['APG', 'FU', 'Week_Year', 'Qty_ordered']
    # df_trainset = pd.read_csv(ds1_ic_orders_csv, low_memory=True)
    cols, df_trainset = load_dataset(ds1_ic_orders_csv)
    print df_trainset
    print df_trainset.shape
    print df_trainset.columns.values
    df_sub_trainset = subset_train_dataset(df_trainset, subcols)
    pickle.dump(df_sub_trainset, open(ds1_sub_ic_orders_pkl, 'wb'))


def load_ds1_ic_orders(ds1_sub_ic_orders_pkl):
    df_trainset = pickle.load(open(ds1_sub_ic_orders_pkl, 'rb'))
    print 'loaded ', ds1_sub_ic_orders_pkl
    print df_trainset.shape


def subset_ds1_ic_orders(ds1_ic_orders_csv, ds1_sub_ic_orders_pkl):
    # cols = ['APG','APG_desc','FU','FU_desc','ana_category','Week_Year','Qty_ordered']
    subcols = ['APG', 'FU', 'Week_Year', 'Qty_ordered']
    # df_trainset = pd.read_csv(ds1_ic_orders_csv, low_memory=True)
    cols, df_trainset = load_dataset(ds1_ic_orders_csv)
    print df_trainset
    print df_trainset.shape
    print df_trainset.columns.values
    df_sub_trainset = subset_train_dataset(df_trainset, subcols)
    pickle.dump(df_sub_trainset, open(ds1_sub_ic_orders_pkl, 'wb'))


def subset_ds1_ic_sales(ds1_ic_sales_csv, ds1_sub_ic_sales_pkl):
    subcols = ['APG_Code',
            'FU',
            'Calendar_Week_Number',
            'Tot_Sls_Unit_Vol_Qt',
            'Less_12_Male',
            'Between_12_17_Male',
            'Between_18_24_Male',
            'Between_25_34_Male',
            'Between_35_44_Male',
            'Between_45_54_Male',
            'Between_55_64_Male',
            'MoreThan_65_Male',
            'Less_12_Female',
            'Between_12_17_Female',
            'Between_18_24_Female',
            'Between_25_34_Female',
            'Between_35_44_Female',
            'Between_45_54_Female',
            'Between_55_64_Female',
            'MoreThan_65_Female',
            'Median_Income']

    # df_trainset = pd.read_csv(ds1_ic_sales_csv, low_memory=True)
    cols, df_trainset = load_dataset(ds1_ic_sales_csv)
    # print df_trainset
    print df_trainset.shape
    print df_trainset.columns.values
    df_sub_trainset = subset_train_dataset(df_trainset, subcols)
    df_sub_trainset.columns = ['APG', 'FU', 'weeknum', 'sales_qty',
                                'Less_12_Male', 'Between_12_17_Male', 'Between_18_24_Male',
                                'Between_25_34_Male', 'Between_35_44_Male', 'Between_45_54_Male',
                                'Between_55_64_Male', 'MoreThan_65_Male', 'Less_12_Female',
                                'Between_12_17_Female', 'Between_18_24_Female', 'Between_25_34_Female',
                                'Between_35_44_Female', 'Between_45_54_Female', 'Between_55_64_Female',
                                'MoreThan_65_Female', 'Median_Income']
    pickle.dump(df_sub_trainset, open(ds1_sub_ic_sales_pkl, 'wb'))


def load_ds1_ic_sales(ds1_sub_ic_sales_pkl):
    df_trainset = pickle.load(open(ds1_sub_ic_sales_pkl, 'rb'))
    print 'loaded ', ds1_sub_ic_sales_pkl
    print df_trainset.shape
    # print df_trainset
    print df_trainset.columns.values


def make_sales_by_year(ds1_sub_ic_sales_pkl, ic_sales_by_year_pkl):
    df_sales = pickle.load(open(ds1_sub_ic_sales_pkl, 'rb'))
    sales_by_year = OrderedDict()
    for index, row in df_sales.iterrows():
        apg = row['APG']
        fu = row['FU']
        weeknum = row['weeknum']
        year = str(weeknum)[:4]
        sales = row['sales_qty']
        if apg not in sales_by_year:
            sales_by_year[apg] = OrderedDict()
        if fu not in sales_by_year[apg]:
            sales_by_year[apg][fu] = OrderedDict()
        if year not in sales_by_year[apg][fu]:
            sales_by_year[apg][fu][year] = OrderedDict()
        sales_by_year[apg][fu][year][weeknum] = sales
    pickle.dump(sales_by_year, open(ic_sales_by_year_pkl, 'wb'))


def vectorize(df_trainset):
    nn = 0; kk = 0
    null_sales = OrderedDict()
    yearly_sales = OrderedDict()
    sales_by_year = OrderedDict()
    yearly_orders = OrderedDict()
    for index, row in df_trainset.iterrows():
        nn = nn + 1
        if nn % 1000 == 0:
            print ' ... ', nn
        apg = row['APG_Code']
        fu = row['FU']
        weeknum = row['year_week']
        population_12 = row['Less_12_Male'] + row['Less_12_Female']
        population_44 = row['Between_35_44_Male'] + row['Between_35_44_Female']
        population_54 = row['Between_45_54_Male'] + row['Between_45_54_Female']
        sales_qty = row['Tot_Sls_Unit_Vol_Qt']
        orders_qty = row['Qty_ordered']
        if apg not in null_sales:
            null_sales[apg] = OrderedDict()
        if fu not in null_sales[apg]:
            null_sales[apg][fu] = OrderedDict()
        null_sales[apg][fu][weeknum] = sales_qty

        year = str(weeknum)[:4]
        if apg not in yearly_sales:
            yearly_sales[apg] = OrderedDict()
        if fu not in yearly_sales[apg]:
            yearly_sales[apg][fu] = OrderedDict()
        if year not in yearly_sales[apg][fu]:
            yearly_sales[apg][fu][year] = 0
        yearly_sales[apg][fu][year] = yearly_sales[apg][fu][year] + sales_qty

        if apg not in sales_by_year:
            sales_by_year[apg] = OrderedDict()
        if fu not in sales_by_year[apg]:
            sales_by_year[apg][fu] = OrderedDict()
        if year not in sales_by_year[apg][fu]:
            sales_by_year[apg][fu][year] = OrderedDict()
        sales_by_year[apg][fu][year][weeknum] = sales_qty

        if apg not in yearly_orders:
            yearly_orders[apg] = OrderedDict()
        if fu not in yearly_orders[apg]:
            yearly_orders[apg][fu] = OrderedDict()
        if year not in yearly_orders[apg][fu]:
            yearly_orders[apg][fu][year] = 0
        yearly_orders[apg][fu][year] = yearly_orders[apg][fu][year] + orders_qty

        if math.isnan(sales_qty):
            kk = kk + 1
            # print ' *** ', ordered_qty

    # missing data treatment
    for apg in sales_by_year:
        for fu in sales_by_year[apg]:
            for year in sales_by_year[apg][fu]:
                sales_by_year[apg][fu][year] = randomize_missing(apg, fu, year, sales_by_year[apg][fu][year])

    print 'total read: ', nn
    print '# null: ', kk
    print '# non-null', nn - kk
    null_count = OrderedDict()
    for apg in null_sales:
        count = 0
        for fu in null_sales[apg]:
            for weeknum in null_sales[apg][fu]:
                if math.isnan(null_sales[apg][fu][weeknum]):
                    count = count + 1
        null_count[apg] = count
        print apg
        print '\t', null_sales[apg]

    print "======== Fu counts ======="
    for apg in null_sales:
        nfuweeks = sum(len(v) for v in null_sales[apg].itervalues())
        nnull = null_count[apg]
        percent_null = float(nnull) * 100.0 / float(nfuweeks)
        print '%s FU count: %d -- all: %d  -- Percent null: %7.2f%%' % (apg, len(null_sales[apg]), nfuweeks, percent_null)
    print "======== null counts ======="
    for apg in null_count:
        print '%s null count: %d' % (apg, null_count[apg])
    # dump_yearly_sales_orders(yearly_sales, yearly_orders)
    nfu = 0
    nfu3year = 0
    feasible = OrderedDict()
    valid_years = ['2016', '2017', '2018']
    for apg in yearly_sales:
        for fu in yearly_sales[apg]:
            nfu = nfu + 1
            vyr = OrderedDict()
            for year in yearly_sales[apg][fu]:
                if not math.isnan(yearly_sales[apg][fu][year]):
                    vyr[year] = True
            nyr = 0
            for yr in valid_years:
                if yr in vyr:
                    nyr = nyr + 1
            if nyr > 2:
                print '%s\t%s #%d' % (apg, fu, len(yearly_sales[apg][fu]))
                nfu3year = nfu3year + 1
                if apg not in feasible:
                    feasible[apg] = OrderedDict()
                if fu not in feasible[apg]:
                    feasible[apg][fu] = OrderedDict()
                for year in yearly_sales[apg][fu]:
                    if not math.isnan(yearly_sales[apg][fu][year]):
                        feasible[apg][fu][year] = yearly_sales[apg][fu][year]
    print '# Sales FU: ', nfu
    print '# Sales FU with 3 Years data: ', nfu3year
    print '% Sales satisfied: ', float(nfu3year) * 100.0 / float(nfu)

    g = open(train_folder_20180803 + 'ic_ds1_feasible.csv', 'w')
    g.write("APG,FU,YEAR,SALES\n")
    for apg in feasible:
        # g.write('%s\t#FU: %d\n' % (apg, len(feasible[apg])))
        print '%s\t#FU: %d' % (apg, len(feasible[apg]))
        for fu in feasible[apg]:
            # g.write('\t%s\n' % fu)
            print '\t', fu
            for year in feasible[apg][fu]:
                g.write('%s,%s,%s,%d\n' % (apg, fu, year, int(feasible[apg][fu][year])))
                # g.write('\t\t%s\t%d\n' % (year, int(feasible[apg][fu][year])))
                print '\t\t%s\t%9.0f' % (year, feasible[apg][fu][year])
    g.close()

    # nfu = 0
    # nfu3year = 0
    # for apg in yearly_orders:
    #     for fu in yearly_orders[apg]:
    #         nfu = nfu + 1
    #         if len(yearly_orders[apg][fu]) > 2:
    #             print '%s\t%s' % (apg, fu)
    #             nfu3year = nfu3year + 1
    # print '# Orders FU: ', nfu
    # print '# Orders FU with 3 Years data: ', nfu3year
    # print '% Orders satisfied: ', float(nfu3year) * 100.0 / float(nfu)


def arima_predictor(df_endo, df_exo):
    pass


def make_time_series_df(ts):
    df_ts = pd.DataFrame.from_dict(ts)
    return df_ts


def select_timeseries(apg, fu, reg_sales_by_year):
    ts = OrderedDict()
    for year in reg_sales_by_year[apg][fu]:
        for weeknum in reg_sales_by_year[apg][fu][year]:
            ts[weeknum] = reg_sales_by_year[apg][fu][year][weeknum]
    return ts


def vectorize_sales(reg_sales_by_year_pkl):
    sales_by_year = pickle.load(open(reg_sales_by_year_pkl, 'rb'))
    ninst = sum(len(v) for v in sales_by_year.itervalues())
    print '# instances: ', ninst
    kk = 0
    fu_set = OrderedDict()
    apg_set = set()
    for apg in sales_by_year:
        apg_set.add(apg)
        if apg not in fu_set:
            fu_set[apg] = set()
        for fu in sales_by_year[apg]:
            fu_set[apg].add(fu)
            for year in sales_by_year[apg][fu]:
                for weeknum in sales_by_year[apg][fu][year]:
                    kk = kk + 1

    print '# confirmation: ', kk
    print 'APG set: ', apg_set
    for apg in fu_set:
        fu_list = sorted(list(fu_set[apg]))
        print apg
        print '\t# in FU set: ', len(fu_set[apg])
        print '\tFU set: ', fu_set[apg]
        print '\tFU list: ', fu_list

    apg = 'UGB005'
    fu = 'IGB4165'
    ts = select_timeseries(apg, fu, sales_by_year)
    print '\n ~~~~~~~ %s - %s Timeseries ~~~~~~~' % (apg, fu)
    for weeknum in ts:
        print weeknum, ': ', ts[weeknum]

    df_ts = make_time_series_df(ts)
    print df_ts


def main():
    # check_training()
    # colnames, df = load_train_dataset('ic_sorted_ds1')
    # print df
    # print colnames
    # print '# features: ', len(colnames)
    # df_train = subset_train_dataset(df, feature_selection['basic_sans_promo'])
    # print df_train.shape
    # print df_train.columns.values
    # store_trainset(df_train, trainset['ic_ds1_sub1'])

    # df_trainset =load_trainset(trainset['ic_ds1_sub1'])
    # print df_trainset.shape
    # print df_trainset.columns.values

    # check_data(df_trainset)
    # vectorize(df_trainset)
    # analyze_epos_data(epos_data_csv, epos_sales_by_year_pkl)

    # extrapolate_epos_data(epos_sales_by_year_pkl, epos_reguralized_sales_by_year_pkl)
    # eops_regularized_tocsv(epos_reguralized_sales_by_year_pkl, epos_reguralized_sales_by_year_csv)

    # subset_ds1_ic_orders(trainset2['ds1_ic_orders'], trainset2['ds1_sub_ic_orders'])
    # load_ds1_ic_orders(trainset2['ds1_sub_ic_orders'])

    # subset_ds1_ic_sales(trainset2['ds1_ic_sales'], trainset2['ds1_sub_ic_sales'])
    # load_ds1_ic_sales(trainset2['ds1_sub_ic_sales'])
    #
    # make_sales_by_year(trainset2['ds1_sub_ic_sales'], trainset2['ic_sales_by_year'])
    #
    # extrapolate_epos_data(trainset2['ic_sales_by_year'], trainset2['ic_regularized_sales_by_year'])
    # eops_regularized_tocsv(trainset2['ic_regularized_sales_by_year'], trainset2['ic_regularized_sales_csv'])

    vectorize_sales(trainset2['ic_regularized_sales_by_year'])


if __name__ == '__main__':
    print 'predictor started ...'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    main()
    # eg_MLP()
