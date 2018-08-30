
import os
import pandas as pd
import csv
from collections import OrderedDict
import pickle
import pprint

import numpy as np
# from sklearn import datasets, linear_model
# from sklearn.metrics import mean_squared_error, r2_score


data_root_dir = '/Users/jeanclaudelemoyne/work/Data/'
unilever_folder = data_root_dir + 'Unilever/'
unilever_training_folder = data_root_dir + 'Unilever/train/'

data_source_2015 = unilever_folder + 'NielsenTea2015.xlsx'
data_source_2016 = unilever_folder + 'NielsenTea2016.xlsx'
data_source_2017_2018 = unilever_folder + 'NielsenTea2017&2018.xlsx'

train_source_2015 = unilever_training_folder + 'NielsenTea2015.pkl'
train_source_2016 = unilever_training_folder + 'NielsenTea2016.pkl'
train_source_2017_2018 = unilever_training_folder + 'NielsenTea2017&2018.pkl'

train_csv_2015 = unilever_training_folder + 'NielsenTea2015.csv'
train_csv_2016 = unilever_training_folder + 'NielsenTea2016.csv'
train_csv_2017_2018 = unilever_training_folder + 'NielsenTea2017&2018.csv'
train_weka_csv_2017_2018 = unilever_training_folder + 'NielsenTea2017&2018W.csv'
store_master_data = unilever_folder + 'store_master.csv'
uk_demographics = unilever_folder + 'exogenous/ukdetailedtimeseries2001to2017/MYEB1_detailed_population_estimates_series_UK_(0117).csv'
uk_demographics2 = unilever_folder + 'exogenous/ukdetailedtimeseries2001to2017/MYEB2_detailed_components_of_change_series_EW_(0217).csv'
uk_postal_code_to_lad = unilever_folder + 'exogenous/pcd11_par11_wd11_lad11_ew_lu.csv'
uk_postalad_index_pkl = unilever_folder + 'exogenous/postalad_index.pkl'
uk_demographics_pkl = unilever_folder + 'exogenous/demographics.pkl'
uk_demographics_2018_pkl = unilever_folder + 'exogenous/demographics_2018.pkl'
uk_store_demographics_pkl = unilever_folder + 'exogenous/store_demographics.pkl'
uk_store_demographics_csv = unilever_folder + 'exogenous/store_demographics.csv'
uk_store_demographics_2016_2018_csv = unilever_folder + 'exogenous/store_demographics_2016_2018.csv'
store_demo_stats_report = unilever_folder + 'exogenous/store_demo_stats.txt'
store_stats_profile_pkl = unilever_folder + 'exogenous/store_demo_stats.pkl'
uk_store_sim_demographics_pkl = unilever_folder + 'exogenous/store_sim_demographics.pkl'
uk_store_sim_demographics_2016_2018_csv = unilever_folder + 'exogenous/store_sim_demographics_2016_2018.csv'
uk_household_stats_xlsx = unilever_folder + 'exogenous/hdiireferencetables201617.xlsx'
house_hold_income_folder = '/Users/jeanclaudelemoyne/work/Data/Unilever/exogenous/hh_income/'

Retailer_id_toAPG = {
    '504':	'UGB003',
    '525': 	'UGB005',
    '506':	'UGB004',
    '518':	'UGB002',
    '503':	'UGB001'
}


def read_xcel(xcel_fname, pkl_fname):
    ss = pd.read_excel(xcel_fname)
    if pkl_fname is not None:
        pickle.dump(ss, open(pkl_fname, 'wb'))
    print ss


def readExelFile(xcel_fname, hh_folder):
    xls = pd.ExcelFile(xcel_fname)
    sheet_names = xls.sheet_names
    print sheet_names
    for book in sheet_names:
        print ' ========= ', book, ' ========='
        df = pd.read_excel(xls, sheetname=book)
        csv_fname = hh_folder + book + '.csv'
        df.to_csv(csv_fname, header=True, index=False, encoding='utf-8')
        print df


def load_source(pkl_fname):
    ss = pickle.load(open(pkl_fname, 'rb'))
    # print ss.head()
    print list(ss.columns.values)
    print ss.shape
    return ss


def weka_csv_compliance(csv_fname, weka_csv_fname):
    n = 0; eolc = 0
    outsep = ','
    g = open(weka_csv_fname, 'w')
    with open(csv_fname, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter='\t')
        for row in reader:
            if n == 0:
                hdr = row
                hdr = [ '"' + elt + '"' for elt in row]
                g.write(outsep.join(hdr) + '\n')
                n = n + 1
                continue
            if n > 2:
                g.write(outsep.join(row) + '\n')
            k = len(row)
            print n, k
            n = n + 1
            if k != 91:
                print "**** "
                break
            for elt in row:
                if '\n' in elt:
                    eolc = eolc + 1

        print hdr
        print 'hdr len: ', len(hdr)
        print '#eolc: ', eolc

        g.close()


def check_for_dups():
    cols = ["1","""Market Desc""","""TOTAL""","""CATEGORY GROUP""","""CATEGORY""","""MARKET""","""COMPANY""","""GLOBAL BRAND""","""BRAND""","""SEGMENT""","""FORM""","""SUBBRAND""","""SUB SEGMENT""","""FLAVOUR""","""ITEM""","""EAN""","""GB ALL SPECIAL OFFERS""","""GB UNILEVER SEGMENT""","""GB CAFFEINATED VS DECAFFEINATED""","""GB UNILEVER FORM""","""GB FLAVOUR STYLE OF TEA""","""GB PROCESSING WEIGHT""","""ACTUAL NUMBER IN PACK""","""Metrics""","""Value Sales""","""Value Sales.1""","""Value Sales.2""","""Value Sales.3""","""Value Sales.4""","""Value Sales.5""","""Value Sales.6""","""Value Sales.7""","""Value Sales.8""","""Value Sales.9""","""Value Sales.10""","""Value Sales.11""","""Value Sales.12""","""Value Sales.13""","""Value Sales.14""","""Value Sales.15""","""Value Sales.16""","""Volume Sales""","""Volume Sales.1""","""Volume Sales.2""","""Volume Sales.3""","""Volume Sales.4""","""Volume Sales.5""","""Volume Sales.6""","""Volume Sales.7""","""Volume Sales.8""","""Volume Sales.9""","""Volume Sales.10""","""Volume Sales.11""","""Volume Sales.12""","""Volume Sales.13""","""Volume Sales.14""","""Volume Sales.15""","""Volume Sales.16""","""Volume Sales Eq2""","""Volume Sales Eq2.1""","""Volume Sales Eq2.2""","""Volume Sales Eq2.3""","""Volume Sales Eq2.4""","""Volume Sales Eq2.5""","""Volume Sales Eq2.6""","""Volume Sales Eq2.7""","""Volume Sales Eq2.8""","""Volume Sales Eq2.9""","""Volume Sales Eq2.10""","""Volume Sales Eq2.11""","""Volume Sales Eq2.12""","""Volume Sales Eq2.13""","""Volume Sales Eq2.14""","""Volume Sales Eq2.15""","""Volume Sales Eq2.16""","""Unit Sales""","""Unit Sales.1""","""Unit Sales.2""","""Unit Sales.3""","""Unit Sales.4""","""Unit Sales.5""","""Unit Sales.6""","""Unit Sales.7""","""Unit Sales.8""","""Unit Sales.9""","""Unit Sales.10""","""Unit Sales.11""","""Unit Sales.12""","""Unit Sales.13""","""Unit Sales.14""","""Unit Sales.15""","""Unit Sales.16"""]
    print '# cols: ', len(cols)
    attrset = set()
    for col in cols:
        attrset.add(col)
    print '# attr: ', len(attrset)


def load_portal_to_lad_index(map_pcd_lad_csv_fname, postalad_pkl):
    postalToLad = OrderedDict()
    wardT0lad = OrderedDict()
    force = False
    if not force and os.path.exists(postalad_pkl):
        wardToLad, postalToLad = pickle.load(open(postalad_pkl, 'rb'))
    else:
        dfpcd = pd.read_csv(map_pcd_lad_csv_fname, low_memory=False)
        print 'postal to lad shape: ', dfpcd.shape
        for index, row in dfpcd.iterrows():
            if index % 10000 == 0:
                print '... ', index
            pcd = row['pcd7']
            if len(pcd) == 6:
                district = pcd[:2]
            elif len(pcd) == 7:
                district = pcd[:3]
            elif len(pcd) == 8:
                district = pcd[:4]
            else:
                district = pcd[:3]

            lad = row['lad11cd']
            postalToLad[district] = lad
            ward = row['wd11nm'].upper()
            wardT0lad[ward] = lad
        pickle.dump((wardT0lad, postalToLad), open(postalad_pkl, 'wb'))
        print postalToLad
    return wardT0lad, postalToLad


def get_age_group(age):
    age_index = [12, 18, 25, 35, 45, 55, 65, 120]
    age_group = OrderedDict()
    age_group[12] = 'less-12'
    age_group[18] = '12-17'
    age_group[25] = '18-24'
    age_group[35] = '25-34'
    age_group[45] = '35-44'
    age_group[55] = '45-54'
    age_group[65] = '55-64'
    age_group[120] = 'more-65'

    for aix in age_index:
        if age < aix:
            group = aix
            break

    return age, group, age_group[group]


def init_age_group():
    gender_age_group = OrderedDict()
    # 1 = Male 2 = Female
    age_group_labels = ['less-12', '12-17', '18-24', '25-34', '35-44', '45-54', '55-64', 'more-65']
    for year in range(2001, 2018):
        gender_age_group[year] = OrderedDict()
        for gender in range(1, 3):
            gender_age_group[year][gender] = OrderedDict()
            for group_label in age_group_labels:
                gender_age_group[year][gender][group_label] = 0
    return gender_age_group


def zero_clear_age_groups(gender_age_group):
    for year in gender_age_group:
        for gender in gender_age_group[year]:
            for group in gender_age_group[year][gender]:
                gender_age_group[year][gender][group] = 0
    return gender_age_group


def update_age_group_counts(year, gender, age, count, gender_age_group):
    _, _, group = get_age_group(age)
    gender_age_group[year][gender][group] = gender_age_group[year][gender][group] + count
    return gender_age_group


def load_demographics(demographics_pkl):
    demographics = pickle.load(open(demographics_pkl, 'rb'))
    print len(demographics)
    # nn = 0
    # for lad in demographics:
    #     nn = nn + 1
    #     for gender in demographics[lad]:
    #         for group in demographics[lad][gender]:
    #             print lad, gender, group, demographics[lad][gender][group]
    #     if nn > 5:
    #         break


def get_store_demographics(store_master_csv_fname, demog_csv_fname,
                            map_pcd_lad_csv_fname, postalad_pkl, demographics_pkl):
    dfstore = pd.read_csv(store_master_csv_fname, low_memory=False)
    postal_codes = dfstore['Location_Postal_Identifier'].values
    districts = dfstore['Location_City_Name'].values
    print postal_codes
    print districts
    postal_code_set = set(postal_codes)
    district_set = set(districts)

    print '# of postal codes: ', len(postal_code_set)
    print '# of cities: ', len(district_set)

    gender_age_group = init_age_group()

    demographics = OrderedDict()

    dfdemo = pd.read_csv(demog_csv_fname)
    print 'demographics shape: ', dfdemo.shape
    prev_lad_code = None
    for index, row in dfdemo.iterrows():
        if index % 1000 == 0:
            print '... ', index
        lad_code = row['lad2014_code']
        gender_code = row['sex']
        age = row['age']
        if prev_lad_code is None or prev_lad_code is not lad_code:
            if prev_lad_code is not None:
                demographics[prev_lad_code] = OrderedDict()
                for year in range(2001, 2018):
                    demographics[prev_lad_code][year] = OrderedDict()
                    for gender in gender_age_group[year]:
                        demographics[prev_lad_code][year][gender] = OrderedDict()
                        for group in gender_age_group[year][gender]:
                            demographics[prev_lad_code][year][gender][group] = gender_age_group[year][gender][group]
            prev_lad_code = lad_code
            gender_age_group = zero_clear_age_groups(gender_age_group)
            demographics[lad_code] = OrderedDict()
            for year in range(2001, 2018):
                demographics[lad_code][year] = OrderedDict()

        for year in range(2001, 2018):
            col_name = 'population_' + str(year)
            population= row[col_name]
            gender_age_group = update_age_group_counts(year, gender_code, age, population, gender_age_group)

    for year in range(2001, 2018):
        for gender in gender_age_group[year]:
            demographics[lad_code][year][gender] = OrderedDict()
            for group in gender_age_group[year][gender]:
                demographics[lad_code][year][gender][group] = gender_age_group[year][gender][group]

    pickle.dump(demographics, open(demographics_pkl, 'wb'))

    localities = dfdemo['lad2014_name']
    locality_set = set(localities)
    print localities
    print '$ localities:', len(locality_set)

    # postalToLad = load_portal_to_lad_index(map_pcd_lad_csv_fname, postalad_pkl)
    # print '# postalad: ', len(postalToLad)


def dump_demograhics(demographics_pkl):
    demographics = pickle.load(open(demographics_pkl, 'rb'))
    for lad in demographics:
        for year in demographics[lad]:
            for gender in demographics[lad][year]:
                for group in demographics[lad][year][gender]:
                    print lad, year, 'gender = ', gender, group, demographics[lad][year][gender][group]


def dump_store_demograhics(store_demographics_pkl):
    store_demo = pickle.load(open(store_demographics_pkl, 'rb'))
    for store_id in store_demo:
        print store_id, store_demo[store_id]
        for year in store_demo[store_id]:
            for gender in store_demo[store_id][year]:
                for group in store_demo[store_id][year][gender]:
                    print store_id, gender, year, group, store_demo[store_id][year][gender][group]
    print '# store id: ', len(store_demo)


def estimate_population(X, Y, projected_year):
    x = np.array(X)
    y = np.array(Y)
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return int(m * projected_year + c)


def extrapolate_demographics(demographics_pkl, projected_year, demographics_2018_pkl):
    demographics = pickle.load(open(demographics_pkl, 'rb'))
    age_group_labels = ['less-12', '12-17', '18-24', '25-34', '35-44', '45-54', '55-64', 'more-65']
    for lad in demographics:
        demographics[lad][projected_year] = OrderedDict()
        for gender in range(1, 3):
            demographics[lad][projected_year][gender] = OrderedDict()
            X = []
            Y = []
            for group in age_group_labels:
                for year in range(2001, 2018):
                    X = X + [year]
                    Y = Y + [demographics[lad][year][gender][group]]
                    # print lad, gender, year, group, demographics[lad][year][gender][group]
                estimated = estimate_population(X, Y, projected_year)
                demographics[lad][projected_year][gender][group] = estimated
                print lad, gender, group, 'projection for 2018: ', estimated

    pickle.dump(demographics, open(demographics_2018_pkl, 'wb'))


def store_demo_stats(store_demographics_pkl, stats_fname, stats_profile_pkl):
    group_obs = OrderedDict()
    store_demo = pickle.load(open(store_demographics_pkl, 'rb'))
    print 'reading obs ...'
    nn = 0
    for store_id in store_demo:
        nn = nn + 1
        if nn % 200 == 0:
            print ' ... ', nn
        for year in store_demo[store_id]:
            if year not in group_obs:
                group_obs[year] = OrderedDict()
            for gender in store_demo[store_id][year]:
                if gender not in group_obs[year]:
                    group_obs[year][gender] = OrderedDict()
                for group in store_demo[store_id][year][gender]:
                    obs = store_demo[store_id][year][gender][group]
                    if group not in group_obs[year][gender]:
                        group_obs[year][gender][group] = [obs]
                    else:
                        group_obs[year][gender][group] = group_obs[year][gender][group] + [obs]
                    # print store_id, gender, year, group, obs

    print '# stores: ', len(store_demo)
    g = open(stats_fname, 'w')
    print 'calculating stats ...'
    print 'Demographics stats for Unilever stores and for 5 APGs'
    g.write('Demographics stats for Unilever stores and for 5 APGs\n')
    stats = OrderedDict()
    for year in range(2016, 2019):
        print 'year: ', year
        g.write('year: ' + str(year) + '\n')
        stats[year] = OrderedDict()
        for gender in group_obs[year]:
            print '\tgender: ', gender
            g.write('\tgender: ' + str(gender) + '\n')
            stats[year][gender] = OrderedDict()
            for group in group_obs[year][gender]:
                print '\t\t' + group
                g.write('\t\t' + group + '\n')
                x = np.array(group_obs[year][gender][group])
                stats[year][gender][group] = OrderedDict()
                stats[year][gender][group]['median'] = np.median(x)
                stats[year][gender][group]['mean'] = np.mean(x)
                stats[year][gender][group]['std'] = np.std(x)
                print '\t\t\tmedian: ', stats[year][gender][group]['median']
                print '\t\t\tmean: ', stats[year][gender][group]['mean']
                print '\t\t\tstd dev: ', stats[year][gender][group]['std']
                g.write('\t\t\tmedian: ' + str(int(stats[year][gender][group]['median'])) + '\n')
                g.write('\t\t\tmean: ' + str(int(stats[year][gender][group]['mean'])) + '\n')
                g.write('\t\t\tstd dev: ' + str(int(stats[year][gender][group]['std'])) + '\n')
    g.close()
    pickle.dump(stats, open(stats_profile_pkl, 'wb'))


def load_store_demo_stats(stats_profile_pkl):
    print stats_profile_pkl
    demo_stats = pickle.load(open(stats_profile_pkl, 'rb'))
    return demo_stats


def show_demo_stats(demo_stats):
    for year in range(2016, 2019):
        print 'year: ', year
        for gender in demo_stats[year]:
            print '\tgender: ', gender
            for group in demo_stats[year][gender]:
                print '\t\t' + group
                print '\t\t\tmedian: ', demo_stats[year][gender][group]['median']
                print '\t\t\tmean: ', demo_stats[year][gender][group]['mean']
                print '\t\t\tstd dev: ', demo_stats[year][gender][group]['std']


def simulate_randomize_store_demographics(demo_stats):
    # show_demo_stats(demo_stats)
    demographics = OrderedDict()
    for year in demo_stats:
        demographics[year] = OrderedDict()
        for gender in demo_stats[year]:
            demographics[year][gender] = OrderedDict()
            for group in demo_stats[year][gender]:
                median = demo_stats[year][gender][group]['median']
                mean = demo_stats[year][gender][group]['mean']
                std = demo_stats[year][gender][group]['std']
                population = int(np.random.normal(mean, std))
                if population < 0:
                    population = int(std)
                demographics[year][gender][group] = population
                # print median, mean, std, '  >>>>>> ', population
    return demographics


def simulate_store_demo(store_demographics_pkl, store_master_csv_fname,
                        stats_profile_pkl, store_sim_demographics_pkl):
    store_demo = pickle.load(open(store_demographics_pkl, 'rb'))
    dfstore = pd.read_csv(store_master_csv_fname, low_memory=False)
    demo_stats = load_store_demo_stats(stats_profile_pkl)

    store_demo_ids = set(store_demo.keys())
    store_ids = dfstore['Retailer_Store_Identifier']
    store_id_set = set(store_ids)

    delta = store_id_set.difference(store_demo_ids)

    print '$ store demo ids: ', len(store_demo_ids)
    print '# store_ids: ', len(store_ids), ' # unique: ', len(store_id_set), ' # delta: ', len(delta)

    for stid in delta:
        store_demo[stid] = simulate_randomize_store_demographics(demo_stats)
    pickle.dump(store_demo, open(store_sim_demographics_pkl, 'wb'))
    print ' ... randomized simulation done!'


def neighbor(pcd, postalad, demographics):
    if pcd in postalad and postalad[pcd] in demographics:
        return postalad[pcd]
    if len(pcd) == 3:
        area = pcd[:2]
        for i in range(10):
            di = area + str(i)
            if di in postalad:
                if postalad[di] in demographics:
                    return postalad[di]
    return None


def match_store_demographics(store_master_csv_fname, postalad_pkl, demographics_pkl,
                             store_demographics_pkl):
    demograhics = pickle.load(open(demographics_pkl, 'rb'))
    print '# demographics: ', len(demograhics)
    wardlad, postalad = pickle.load(open(postalad_pkl, 'rb'))
    nn = 0
    for pcd in postalad:
        if pcd.startswith('TF3'):
            print pcd, postalad[pcd]
        nn = nn + 1
        # if nn > 100:
        #     break

    print ' # postal to lad: ', len(postalad)
    dfstore = pd.read_csv(store_master_csv_fname, low_memory=False)
    nn = 0
    kk = 0
    ie = 0
    er = 0
    le = 0
    store_demo = OrderedDict()
    for index, row in dfstore.iterrows():
        store_id = row['Retailer_Store_Identifier']
        pcd = row['Location_Postal_Identifier']
        store_name = row['Unilever_Store_Name']
        if isinstance(store_name, str):
            store_city = store_name.split(' ')
        else:
            store_city = None
        # print ':::::: ', store_city, ' ??? ', store_city in wardlad
        if isinstance(pcd, str):
            district = pcd[:3]
            if district in postalad:
                lad = postalad[district]
                # print pcd, ': TRUE ', lad
                store_demo[store_id] = OrderedDict()
                if lad in demograhics:
                    store_demo[store_id] = demograhics[lad]
                    kk = kk + 1
                elif store_city is not None:
                    for town in store_city:
                        hit = False
                        if town in wardlad and wardlad[town] in demograhics:
                            store_demo[store_id] = demograhics[wardlad[town]]
                            hit = True
                            kk = kk + 1
                            break
                    if not hit:
                        di = neighbor(district, postalad, demograhics)
                        if di is not None:
                            store_demo[store_id] = demograhics[di]
                            kk = kk + 1
                        else:
                            le = le + 1
                            # print le, ' ** 1 ** ========== ????? ', district, ' == ', len(district)
                else:
                    le = le + 1
                    # print le, ' ** 2 ** ========== ????? ', district, ' == ', len(district)
            else:
                if store_city is not None:
                    for town in store_city:
                        hit = False
                        if town in wardlad and wardlad[town] in demograhics:
                            store_demo[store_id] = demograhics[wardlad[town]]
                            hit = True
                            kk = kk + 1
                            break
                    if not hit:
                        di = neighbor(district, postalad, demograhics)
                        if di is not None:
                            store_demo[store_id] = demograhics[di]
                            kk = kk + 1
                        else:
                            le = le + 1
                            # print le, ' ** 3 ** ========== ????? ', district, ' == ', len(district)
                else:
                    # print '******* ', pcd, ': FALSE'
                    ie = ie + 1
        else:
            if store_city is not None:
                for town in store_city:
                    hit = False
                    if town in wardlad and wardlad[town] in demograhics:
                        store_demo[store_id] = demograhics[wardlad[town]]
                        hit = True
                        kk = kk + 1
                        break
                if not hit:
                    di = neighbor(district, postalad, demograhics)
                    if di is not None:
                        store_demo[store_id] = demograhics[di]
                        kk = kk + 1
                    else:
                        le = le + 1
                        # print le, ' ** 4 ** ========== ????? ', district, ' == ', len(district)
            else:
                # print er, '~~~~~~~~~~~ pcd: ', pcd, ' store id:', store_id
                er = er + 1

        nn = nn + 1
        # if nn > 100:
        #     break

    print 'kk: %d\tie: %d\ter: %d\tle: %d\ttotal: %d' % (kk, ie, er, le, nn)
    print 'sum: ', kk + ie
    print ' % match: ', float(kk) / float(nn)
    print 'len store_demo: ', len(store_demo)

    pickle.dump(store_demo, open(store_demographics_pkl, 'wb'))


def store_demo_to_csv(store_demographics_pkl, store_demographics_csv):
    store_demo = pickle.load(open(store_demographics_pkl, 'rb'))
    g = open(store_demographics_csv, 'w')
    nn = 0
    for store_id in store_demo:
        nn = nn + 1
        if nn == 1:
            hdr_list = store_demo[store_id][2001][1].keys()
            hdr = ['store_id']
            for year in range(2016, 2019):
                hdr_M = ['M' + str(year) + '-' + x for x in hdr_list]
                hdr_F = ['F' + str(year) + '-' + x for x in hdr_list]
                hdr = hdr + hdr_M + hdr_F
            g.write(','. join(hdr) + '\n')
        # 1 = M, 2 = F
        if store_id not in store_demo or store_demo[store_id] is None or len(store_demo[store_id]) == 0:
            continue
        row = [store_id]
        for year in range(2016, 2019):
            for gender in store_demo[store_id][year]:
                for group in store_demo[store_id][year][gender]:
                    population = store_demo[store_id][year][gender][group]
                    row = row + [population]
                    # print gender, group, store_demo[store_id][gender][group]
        row = [str(x) for x in row]
        g.write(','.join(row) + '\n')
        # if nn > 5:
        #     break

    g.close()
    print len(hdr), hdr


if __name__ == '__main__':
    if os.path.exists(data_source_2017_2018):
        print '>>> Data from ' + data_source_2017_2018
        if not os.path.exists(unilever_training_folder):
            os.makedirs(unilever_training_folder)

        # read_xcel(data_source_2017_2018, train_source_2017_2018)
        # ssdf_2017_2018 = load_source(train_source_2017_2018)
        # ssdf_2017_2018.to_csv(train_csv_2017_2018, sep='\t', index=False)
        # weka_csv_compliance(train_csv_2017_2018, train_weka_csv_2017_2018)
        # check_for_dups()

        # read_xcel(data_source_2016, train_source_2016)
        # ssdf_2016 = load_source(train_source_2016)

        # read_xcel(data_source_2015, train_source_2015)
        # ssdf_2015 = load_source(train_source_2015)

        # get_store_demographics(store_master_data, uk_demographics2, uk_postal_code_to_lad,
        #                        uk_postalad_index_pkl, uk_demographics_pkl)

        # dump_demograhics(uk_demographics_2018_pkl)

        # extrapolate_demographics(uk_demographics_pkl, 2018, uk_demographics_2018_pkl)

        # load_demographics(uk_demographics_pkl)
        # load_portal_to_lad_index(uk_postal_code_to_lad, uk_postalad_index_pkl)
        # match_store_demographics(store_master_data, uk_postalad_index_pkl, uk_demographics_2018_pkl,
        #                          uk_store_demographics_pkl)

        # store_demo_to_csv(uk_store_demographics_pkl, uk_store_demographics_2016_2018_csv)

        # dump_store_demograhics(uk_store_demographics_pkl)

        # store_demo_stats(uk_store_demographics_pkl, store_demo_stats_report, store_stats_profile_pkl)

        # stats = load_store_demo_stats(store_stats_profile_pkl)
        # simulate_randomize_store_demographics(stats)

        # simulate_store_demo(uk_store_demographics_pkl, store_master_data, store_stats_profile_pkl, uk_store_sim_demographics_pkl)
        # store_demo_to_csv(uk_store_sim_demographics_pkl, uk_store_sim_demographics_2016_2018_csv )

        readExelFile(uk_household_stats_xlsx, house_hold_income_folder)