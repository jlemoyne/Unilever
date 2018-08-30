from pandas import read_csv
from datetime import datetime
from collections import OrderedDict

source_folder = '/Users/jeanclaudelemoyne/work/Data/samples/'
raw_data_csv = source_folder + 'raw.csv'
train_folder = '/Users/jeanclaudelemoyne/work/Data/Unilever/train/20180808/'
ds2_orders_csv = train_folder + 'DS2A_orders_sorted.csv'
ds2_train_orders_csv = train_folder + 'ds2_train_orders.csv'


# load data
def parse(x):
    return datetime.strptime(x, '%Y %m %d %H')


def lstm_eg_data():
    dataset = read_csv(raw_data_csv,  parse_dates = [['year', 'month', 'day', 'hour']], index_col=0, date_parser=parse)
    dataset.drop('No', axis=1, inplace=True)
    # manually specify column names
    dataset.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
    dataset.index.name = 'date'
    # mark all NA values with 0
    dataset['pollution'].fillna(0, inplace=True)
    # drop the first 24 hours
    dataset = dataset[24:]
    # summarize first 5 rows
    print(dataset.head(5))
    # save to file
    dataset.to_csv(source_folder + 'pollution.csv')


def vectorize_orders():
    df_orders = read_csv(ds2_orders_csv)
    df_orders.columns = ['APG', 'FU', 'weeknum', 'orders']
    print df_orders.head()
    print 'df_orders shape: ', df_orders.shape
    counter = OrderedDict()
    dataset = OrderedDict()
    for index, row in df_orders.iterrows():
        apg = row['APG']
        fu = row['FU']
        weeknum = row['weeknum']
        orders = row['orders']
        keycol = apg + '-' + fu
        counter[keycol] = 1
        if weeknum not in dataset:
            dataset[weeknum] = OrderedDict()
        dataset[weeknum][keycol] = orders

    nn = len(counter)
    hdr = ['weeknum'] + counter.keys()
    print '# of cols: ', len(hdr)

    print '# APG x FU: ', nn

    g = open(ds2_train_orders_csv, 'w')
    g.write(','.join(hdr) + '\n')
    for weeknum in dataset:
        counter.clear()
        for keycol in dataset[weeknum]:
            counter[keycol] = dataset[weeknum][keycol]
        row = [weeknum]
        row = row + counter.values()
        row = [str(x) for x in row]
        g.write(','.join(row) + '\n')
    g.close()


if __name__ == '__main__':
    vectorize_orders()