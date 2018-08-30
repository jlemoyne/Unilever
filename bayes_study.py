#!/usr/bin/env python

'''
    Unilever POC Analysis for Predicting Shipments
    Anaplan July 2018 - Jean Claude Lemoyne
'''

import csv
from collections import OrderedDict
import pickle
import pandas as pd
import numpy as np
import datetime


unilever_folder = '/Users/jeanclaudelemoyne/work/Data/Unilever/'
ic_ship_2016_2018 = '/Users/jeanclaudelemoyne/work/Data/Unilever/ICDressing/ic_ship.csv'
ic_sub_ship_2016_2018 = '/Users/jeanclaudelemoyne/work/Data/Unilever/ICDressing/ic_sub_ship.csv'
train_ic_ship_2016_2018 = '/Users/jeanclaudelemoyne/work/Data/Unilever/ICDressing/train_ic_ship.csv'
train_ic_ship_qty_2016_2018 = '/Users/jeanclaudelemoyne/work/Data/Unilever/ICDressing/train_ic_ship_qty.csv'
ic_ship_seq_freq = '/Users/jeanclaudelemoyne/work/Data/Unilever/ICDressing/ic_ship_freq.csv'
ic_delays_seq_freq = '/Users/jeanclaudelemoyne/work/Data/Unilever/ICDressing/ic_delays_freq.csv'
train_ic_ship_freq_2016_2018 = '/Users/jeanclaudelemoyne/work/Data/Unilever/ICDressing/train_ic_ship_freq.csv'
ship_analysis = '/Users/jeanclaudelemoyne/work/Data/Unilever/ICDressing/results/ship_analysis.txt'
train_root_dir = '/Users/jeanclaudelemoyne/work/Data/Unilever/ICDressing/'
ship_freq_bin = train_root_dir + 'ship_freq.pkl'
train_APG_WEEK_FU_ship_csv = train_root_dir + 'train_apg_week_fu_ship.csv'
epos_apg_fu_csv = unilever_folder + 'EPOS_APG_FU.csv'
orders_apg_fu_csv = unilever_folder + 'Orders_APG_FU.csv'
epos_apg_stores_csv = unilever_folder + 'EPOS_APG_stores.csv'
time_periods_csv = unilever_folder + 'time_periods.csv'
trend_ice_cream_csv = unilever_folder + 'trend_ice_cream_uk.csv'
trend_ice_cream_weekly_csv = unilever_folder + 'trend_ice_cream_weekly_uk.csv'
trend_tea_csv = unilever_folder + 'trend_tea_uk.csv'
trend_tea_weekly_csv = unilever_folder + 'trend_tea_weekly_uk.csv'
orders_ds1_csv = unilever_folder + 'orders_ds1.csv'
orders_ds2_csv = unilever_folder + 'orders_ds2.csv'


def conditional_proba(csv_fname):
    n = 0
    ts = OrderedDict()
    with open(csv_fname, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            if n == 0:
                hdr = row
                n = n + 1
                continue

            # print row

            n = n + 1
            rowhash = dict(zip(hdr, row))
            ts[rowhash['WEEK']] = rowhash['SHPMT']
        # print rowhash
    # pp = pprint.PrettyPrinter(indent=4)
    # pp.pprint(ts)
    print '# weeks: ', len(ts)
    for week in ts:
        print week, ts[week]


def next_item(list, i):
    i = i + 1
    if i < len(list):
        return i, list[i]
    else:
        return -1, None


def next_non_zero(list, i):
    i, x = next_item(list, i)
    while x == 0 and i < len(list):
        i, x = next_item(list, i)
    return i, x


def next_zero(list, i):
    i, x = next_item(list, i)
    while x > 0 and i < len(list):
        i, x = next_item(list, i)
    return i, x


def last_non_zero_index(list):
    i = len(list) - 1
    while i > -1 and list[i] == 0:
        i = i - 1
    return i


def sub_ship(csv_fname):
    df = pd.read_csv(csv_fname)
    print df.shape
    selection = ['UGB001', 'UGB002', 'UGB003', 'UGB004', 'UGB005']
    sub = df.loc[df['APG'].isin(selection)]
    print sub.shape


def bayes_analyze_ship_data(csv_fname, train_fname='train_ship_study.csv', max_lag=8, decision=True):
    df = pd.read_csv(csv_fname)
    print df
    shpmt = df["SHPMT"].values
    print '~~~ len shpmt: ', len(shpmt)
    print '# non zero: ', sum(1 for x in shpmt if x > 0.0)

    i = -1
    i, x = next_non_zero(shpmt, i)
    print '1st index non zero: ', i, x
    nz = last_non_zero_index(shpmt)
    print 'last non-zero index: ', nz

    # backtrack 8 weeks
    g = open(train_fname, 'w')
    hdr = ""
    for ncol in range(max_lag, 0, -1):
        hdr = hdr + '"w-' + str(ncol) + '",'
    hdr = hdr + '"ship"\n'
    # hdr = '"w-8","w-7","w-6","w-5","w-4","w-3","w-2","w-1","ship"\n'
    g.write(hdr)
    nn = 0
    while (i < nz):
        x = shpmt[i]
        inst = []
        for k in range(max_lag, 0, -1):
            j = i - k
            # print '==> [%d]:: %d %f' % (j, k, shpmt[j])
            inst += [shpmt[j]]

        if decision:
            if x > 0:
                inst += ['S']
            else:
                inst += ['N']
        else:
            inst += [x]

        nn = nn + 1
        # print nn, ' ~~~>> ', inst
        strinst = ['"' + str(x) + '"' for x in inst]
        g.write(','.join(strinst) + '\n')
        i = i + 1
    g.close()


def montecarlo_analyze_ship_data(csv_fname, freq_fname):
    df = pd.read_csv(csv_fname)
    print df
    shpmt = df["SHPMT"].values
    print '~~~ len shpmt: ', len(shpmt)
    print '# non zero: ', sum(1 for x in shpmt if x > 0.0)

    conship = OrderedDict()
    for seq in range(1, 33):
        conship[seq] = OrderedDict()
        conship[seq]['freq'] = 0
        conship[seq]['cumul'] = 0
        conship[seq]['BPr'] = 0.0
        conship[seq]['Prior'] = 0.0

    i = -1
    nz = last_non_zero_index(shpmt)
    while i < nz:
        i, x = next_non_zero(shpmt, i)
        print '1st index non zero: ', i, x
        print 'last non-zero index: ', nz
        k, z = next_zero(shpmt, i)
        seq = k - i
        conship[seq]['freq'] = conship[seq]['freq'] + 1
        print i, x, k, z, seq
        i = k

    sum_freq = 0
    for seq in conship:
        sum_freq = sum_freq + conship[seq]['freq']

    print 'sum freq: %d' % sum_freq

    conship[1]['cumul'] = conship[1]['freq']
    for seq in range(2, len(conship) + 1):
        conship[seq]['cumul'] = conship[seq - 1]['cumul'] + conship[seq]['freq']

    conship[1]['BPr'] = float(conship[1]['cumul']) / float(sum_freq)
    conship[1]['Prior'] = float(conship[1]['cumul']) / float(sum_freq)
    for seq in range(2, len(conship)):
        conship[seq]['BPr'] = float(conship[seq]['freq']) / float(conship[seq]['cumul'])
        conship[seq]['Prior'] = float(conship[seq]['freq']) / float(sum_freq)

    g = open(freq_fname, 'w')
    g.write('"seq","freq","cumul,"BPr"\n')
    for seq in conship:
        print 'seq:%d freq = %d cumul = %d BPr = %f Prior = %f' % (seq, conship[seq]['freq'], conship[seq]['cumul'],
                                                        conship[seq]['BPr'], conship[seq]['Prior'])
        g.write('%d,%d,%d,%f,%f\n' % (seq, conship[seq]['freq'], conship[seq]['cumul'],
                                   conship[seq]['BPr'], conship[seq]['Prior']))
    g.close()


def montecarlo_analyze_delays_data(csv_fname, freq_fname):
    df = pd.read_csv(csv_fname)
    print df
    shpmt = df["SHPMT"].values
    print '~~~ len shpmt: ', len(shpmt)
    print '# non zero: ', sum(1 for x in shpmt if x > 0.0)
    max_delays = 50
    delays = OrderedDict()
    for seq in range(1, max_delays + 1):
        delays[seq] = OrderedDict()
        delays[seq]['freq'] = 0
        delays[seq]['cumul'] = 0
        delays[seq]['BPr'] = 0.0
        delays[seq]['Prior'] = 0.0

    i = -1
    nz = last_non_zero_index(shpmt)
    niter = 0
    while i < nz - 2:
        niter = niter + 1
        i, x = next_zero(shpmt, i)
        print '1st index zero: ', i, x, i < nz
        print 'current index: %d  last non-zero index: %d' % (i, nz)
        k, z = next_non_zero(shpmt, i)
        seq = k - i
        if seq > max_delays:
            seq = max_delays
        if seq < 0:
            seq = max_delays
        delays[seq]['freq'] = delays[seq]['freq'] + 1
        print i, x, k, z, seq
        i = k

    sum_freq = 0
    for seq in delays:
        sum_freq = sum_freq + delays[seq]['freq']

    print 'sum freq: %d' % sum_freq

    delays[1]['cumul'] = delays[1]['freq']
    for seq in range(2, len(delays) + 1):
        delays[seq]['cumul'] = delays[seq - 1]['cumul'] + delays[seq]['freq']

    delays[1]['BPr'] = float(delays[1]['cumul']) / float(sum_freq)
    delays[1]['Prior'] = float(delays[1]['cumul']) / float(sum_freq)
    for seq in range(2, len(delays) + 1):
        delays[seq]['BPr'] = float(delays[seq]['freq']) / float(delays[seq]['cumul'])
        delays[seq]['Prior'] = float(delays[seq]['freq']) / float(sum_freq)

    g = open(freq_fname, 'w')
    g.write('"seq","freq","cumul,"BPr","Prior"\n')
    for seq in delays:
        print 'seq:%d freq = %d cumul = %d BPr = %f Prior = %f' % (seq, delays[seq]['freq'], delays[seq]['cumul'],
                                                        delays[seq]['BPr'], delays[seq]['Prior'])
        g.write('%d,%d,%d,%f,%f\n' % (seq, delays[seq]['freq'], delays[seq]['cumul'],
                                   delays[seq]['BPr'], delays[seq]['Prior']))
    g.close()


'''
    SHIPMENT FREQ ANALYSIS APG by WEEK by FU    

'''


def ship_freq_by_APG_Week(csv_fname, quantities=False, train=train_ic_ship_freq_2016_2018):
    df = pd.read_csv(csv_fname)
    # print df
    freq = OrderedDict()
    fu_sets = OrderedDict()
    apg_freq = OrderedDict()
    nn = 0
    for index, row in df.iterrows():
        nn = nn + 1
        apg = row['APG']
        week = row['Week']
        ship = row['SHPMT']
        fu = row['FU']
        if nn < 5:
            print apg, week, ship

        if ship > 0:
            if apg not in fu_sets:
                fu_sets[apg] = set()
            fu_sets[apg].add(fu)

            if apg not in apg_freq:
                apg_freq[apg] = 0
            apg_freq[apg] = apg_freq[apg] + 1

            if apg not in freq:
                freq[apg] = OrderedDict()
                freq[apg][week] = OrderedDict()
                freq[apg][week]['fu'] = set()
                freq[apg][week]['fu'].add(fu)
                freq[apg][week]['fu_qty'] = set()
                freq[apg][week]['fu_qty'].add((fu, ship))
                if quantities:
                    freq[apg][week]['ship'] = ship
                else:
                    freq[apg][week]['ship'] = 1
            else:
                if week not in freq[apg]:
                    freq[apg][week] = OrderedDict()
                    freq[apg][week]['fu'] = set()
                    freq[apg][week]['fu'].add(fu)
                    freq[apg][week]['fu_qty'] = set()
                    freq[apg][week]['fu_qty'].add((fu, ship))
                    if quantities:
                        freq[apg][week]['ship'] = ship
                    else:
                        freq[apg][week]['ship'] = 1
                else:
                    freq[apg][week]['fu'].add(fu)
                    freq[apg][week]['fu_qty'].add((fu, ship))
                    if quantities:
                        freq[apg][week]['ship'] = freq[apg][week]['ship'] + ship
                    else:
                        freq[apg][week]['ship'] = freq[apg][week]['ship'] + 1

    g = open(train, 'w')
    g.write('"APG","WEEK","QTY"."FU_SET"\n')
    for apg in freq:
        print apg
        for week in freq[apg]:
            ship = freq[apg][week]['ship']
            fu_card = len(freq[apg][week]['fu'])
            print '\t', week, ship, ' #FU: ', fu_card
            row = '"%s","%s","%s","%s"\n' % (apg, week, ship, fu_card)
            g.write(row)
    g.close()

    apgs = fu_sets.keys()
    all_inter = fu_sets[apgs[0]]
    all_union = fu_sets[apgs[0]]
    for i in range(1, len(apgs)):
        all_inter = all_inter.intersection(fu_sets[apgs[i]])
        all_union = all_union.union(fu_sets[apgs[i]])

    pickle.dump((freq, all_union), open(ship_freq_bin, 'wb'))

    res = open(ship_analysis, 'w')
    res.write('Stats for Unilever IC shipments\n')
    print '~~~~~~~~~~~~~~~~~~'
    res.write('APG\t#FU\tPr JACCARD\tSHIP Freq\n')
    for apg in fu_sets:
        PrJaccard = float(len(fu_sets[apg]))/float(len(all_union))
        print apg, len(fu_sets[apg]), PrJaccard, apg_freq[apg]
        res.write('%s\t%d\t%5.2f\t%d\n' % (apg, len(fu_sets[apg]), PrJaccard, apg_freq[apg]))

    res.write('APG\tAPG\tCommon FU Count\n')
    for i in range(len(apgs)):
        for j in range(i + 1, len(apgs)):
            apg_i = apgs[i]
            apg_j = apgs[j]
            inter = fu_sets[apg_i].intersection(fu_sets[apg_j])
            print (i, j), (apg_i, apg_j), ' # inter: ', len(inter)
            res.write('%s\t%s\t%d\n' % (apg_i, apg_j, len(inter)))
    print 'all APG FU union: ', len(all_union)
    print 'all APG FU intersection: ', len(all_inter)
    res.write('all APG FU union: %d\n' % len(all_union))
    res.write('all APG FU intersection: %d\n' % len(all_inter))
    res.close()

    # produce file per APG for AR analysis
    apg = 'UGB002'
    ar_fname = train_root_dir + 'ic_ar_' + apg + '_ship.csv'
    h = open(ar_fname, 'w')
    max_lag = 12
    hdr = ''
    for ncol in range(max_lag, 0, -1):
        hdr = hdr + '"w-' + str(ncol) + '",'
    hdr = hdr + '"SHPMT"\n'
    h.write(hdr)
    shpmt = []
    for week in freq[apg]:
        ship = freq[apg][week]['ship']
        shpmt += [ship]

    for i in range(max_lag, len(shpmt)):
        inst = []
        for k in range(max_lag, 0, -1):
            j = i - k
            inst += [shpmt[j]]
        inst += [shpmt[i]]
        inst = ['"'+ str(x) + '"' for x in inst]
        row = ','.join(inst)
        h.write(row + '\n')
    h.close()


def vectorize(ship_freq_pkl_name, apg_week_fu_ship_csv):
    freq, all_union = pickle.load(open(ship_freq_pkl_name, 'rb'))
    FU_list = sorted(list(all_union))
    FU_ship = OrderedDict((x, 0) for x in FU_list)
    nn = 0
    hdr = ['"APG","WEEK","SHIP"']
    hdr = hdr + ['"' + x + '"' for x in FU_ship.keys()]
    g = open(apg_week_fu_ship_csv, 'w')
    g.write(','.join(hdr) + '\n')
    for apg in freq:
        print apg
        for week in freq[apg]:
            nn = nn + 1
            ship = freq[apg][week]['ship']
            row = [apg, week, ship]
            fu_set = freq[apg][week]['fu_qty']
            print '\t', nn, week, ship, ' FU Set: ', fu_set
            # Clear to zero first
            for fu in FU_ship:
                FU_ship[fu] = 0
            for fu, fu_ship in fu_set:
                FU_ship[fu] = fu_ship
            row = row + FU_ship.values()
            row = ['"' + str(x) + '"' for x in row]
            print row
            g.write(','.join(row) + '\n')
    g.close()
    # print '\n\n\n', all_union
    # print FU_list
    # print len(all_union), len(FU_list)
    # kk = 0
    # for fu in  FU_ship:
    #     print fu, ' ~~~~> ', FU_ship[fu]
    #     kk = kk + 1
    #     if kk > 20:
    #         break


def apg_fu_analysis(epos_apg_fu_csv, orders_apg_fu_csv):
    dfepos = pd.read_csv(epos_apg_fu_csv)
    dforders = pd.read_csv(orders_apg_fu_csv)
    epos = OrderedDict()
    orders = OrderedDict()
    epos_set = set()
    orders_set = set()
    epos_group = OrderedDict()
    orders_group = OrderedDict()
    for index, row in dfepos.iterrows():
        apg = row['APG']
        fu = row['FU']
        fu_count = row['FU_count']
        if apg not in epos:
            epos[apg] = OrderedDict()
        epos_set.add(fu)
        epos[apg][fu] = fu_count
        if apg not in epos_group:
            epos_group[apg] = set()
        epos_group[apg].add(fu)

    for index, row in dforders.iterrows():
        apg = row['APG']
        fu = row['FU']
        fu_count = row['FU_count']
        if apg not in orders:
            orders[apg] = OrderedDict()
        orders_set.add(fu)
        orders[apg][fu] = fu_count
        if apg not in orders_group:
            orders_group[apg] = set()
        orders_group[apg].add(fu)

    xepos = []
    for apg in epos_group:
        xepos = xepos + [len(epos_group[apg])]

    xorders = []
    for apg in orders_group:
        xorders = xorders + [len(orders_group[apg])]

    x_epos = np.array(xepos)
    x_orders = np.array(xorders)

    std_epos = np.std(x_epos)
    std_orders = np.std(x_orders)

    nfu_epos = len(epos_set)
    nfu_orders = len(orders_set)
    unionfu = epos_set.union(orders_set)
    interfu = epos_set.intersection(orders_set)
    nunionfu = len(unionfu)
    ninterfu = len(interfu)
    print '# epos FUs: ', nfu_epos
    print '# orders FUs: ', nfu_orders
    print 'ratio Orders / EPOS ', float(nfu_epos) / float(nfu_orders)
    print 'Jaccard Pr. ', float(ninterfu) / float(nunionfu)
    print 'stdev epos FU count: ', std_epos
    print 'stdev orders FU count: ', std_orders

    for apg in epos_group:
        fu_set1 = orders_group[apg]
        fu_set2 = epos_group[apg]
        fu_inter = fu_set1.intersection(fu_set2)
        fu_union = fu_set1.union(fu_set2)
        print apg, 'Jaccard Prob: ', float(len(fu_inter)) / float(len(fu_union))


def apg_stores_analysis(epos_apg_stores_csv):
    df = pd.read_csv(epos_apg_stores_csv)
    stores = OrderedDict()
    for index, row in df.iterrows():
        apg = row['APG']
        stid = row['Retailer_Store_Identifier']
        if apg not in stores:
            stores[apg] = set()
        stores[apg].add(stid)

    for apg in stores:
        print '# of stores for %s: %d' % (apg, len(stores[apg]))
    apgs = stores.keys()
    for i in range(len(apgs)):
        apg1 = apgs[i]
        for j in range(i + 1, len(apgs)):
            apg2 = apgs[j]
            inter_count = len(stores[apg1].intersection(stores[apg2]))
            print '%s\t%s # common: %d' % (apg1, apg2, inter_count)


def ice_cream_trend(time_periods_csv, trend_ice_cream_csv, trend_ice_cream_weekly_csv, category='Ice-Cream'):
    tpdf = pd.read_csv(time_periods_csv)
    periods = tpdf['Calendar_Week_Number'].tolist()

    print periods

    for time_interval in periods:
        # print time_interval
        r = datetime.datetime.strptime(str(time_interval) + '-0', "%Y%W-%w")
        dtstr = r.strftime("%Y %B").upper()
        dtlist = dtstr.split(' ')
        yr = dtlist[0]
        mt = dtlist[1][:3]
        print yr, mt

    trend = OrderedDict()
    ticdf = pd.read_csv(trend_ice_cream_csv)
    for index, row in ticdf.iterrows():
        q = row['Quarter']
        x = row[category]
        yq = q.split(' ')
        yr = yq[0]
        qr = yq[1]
        if yr in ['2016', '2017', '2018']:
            print yr, qr, x
            if yr not in trend:
                trend[yr] = OrderedDict()
            trend[yr][qr] = x

    for yr in trend:
        for qr in trend[yr]:
            print '~~~> ', yr, qr, trend[yr][qr]

    trend_byweek = OrderedDict()
    for time_interval in periods:
        # print time_interval
        r = datetime.datetime.strptime(str(time_interval) + '-0', "%Y%W-%w")
        dtstr = r.strftime("%Y %B").upper()
        dtlist = dtstr.split(' ')
        yr = dtlist[0]
        mt = dtlist[1][:3]
        print yr, mt
        if mt in trend[yr]:
            prev = trend_byweek[time_interval] = trend[yr][mt]
        else:
            trend_byweek[time_interval] = prev
    g = open(trend_ice_cream_weekly_csv, 'w')
    g.write('weeknum, trend\n')
    for time_interval in trend_byweek:
        x = trend_byweek[time_interval]
        g.write('%d,%s\n' % (time_interval, x))
    g.close()


#     Orders are partitions into with or without EPOS associated to the orders
def analyse_orders_partitions(orders_ds1_csv, orders_ds2_ccsv):
    df1 = pd.read_csv(orders_ds1_csv)
    df2 = pd.read_csv(orders_ds2_ccsv)
    # by Category by APG by FU with FU_desc,Qty_ordered
    ds1 = OrderedDict()
    ds2 = OrderedDict()

    for index, row in df1.iterrows():
        category = row['ana_category']
        apg = row['APG']
        fu = row['FU']
        orders = row['Qty_ordered']
        if category not in ds1:
            ds1[category] = OrderedDict()
        if apg not in ds1[category]:
            ds1[category][apg] = OrderedDict()
        ds1[category][apg][fu] = orders

    for index, row in df2.iterrows():
        category = row['ana_category']
        apg = row['APG']
        fu = row['FU']
        orders = row['Qty_ordered']
        if category not in ds2:
            ds2[category] = OrderedDict()
        if apg not in ds2[category]:
            ds2[category][apg] = OrderedDict()
        ds2[category][apg][fu] = orders

    print '# category in ds1: ', len(ds1)
    for category in ds1:
        print '\t# APG in ds1[%s]: %d' % (category, len(ds1[category]))
    print '# category in ds2: ', len(ds2)
    for category in ds2:
        print '\t# APG in ds2[%s]: %d' % (category, len(ds2[category]))


'''
        M A I N
'''
if __name__ == '__main__':
    print 'Bayes study of shipments ..'
    # conditional_proba('ship_study.csv')
    # bayes_analyze_ship_data('ship_study.csv')

    # sub_ship(ic_ship_2016_2018)

    # bayes_analyze_ship_data(ic_ship_2016_2018, train_ic_ship_2016_2018)
    # bayes_analyze_ship_data(ic_ship_2016_2018, train_ic_ship_qty_2016_2018, max_lag=12, decision=False)

    # montecarlo_analyze_ship_data(ic_sub_ship_2016_2018, ic_ship_seq_freq)
    # montecarlo_analyze_delays_data(ic_sub_ship_2016_2018, ic_delays_seq_freq)

    # ship_freq_by_APG_Week(ic_sub_ship_2016_2018, quantities=True)

    # vectorize(ship_freq_bin, train_APG_WEEK_FU_ship_csv)

    # apg_fu_analysis(epos_apg_fu_csv, orders_apg_fu_csv)

    # apg_stores_analysis(epos_apg_stores_csv)

    # ice_cream_trend(time_periods_csv, trend_ice_cream_csv, trend_ice_cream_weekly_csv)
    # ice_cream_trend(time_periods_csv, trend_tea_csv, trend_tea_weekly_csv, category='Tea')

    analyse_orders_partitions(orders_ds1_csv, orders_ds2_csv)