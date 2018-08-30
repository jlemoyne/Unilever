import logging
import os
# import cloudstorage as gcs
# import webapp2
import pandas as pd
from collections import OrderedDict
import pickle
import numpy as np

'''
SELECT EPOS.*,PHMapping.FG_Code,PHMapping.FG_Description,PHMapping.FU_Description,PHMapping.FU_Code FROM `EPOS_Latest.EPOS_Store_Weather_Item_Mapping` as EPOS
left join `temp.PH_Mapping` as PHMapping on EPOS.Upc_Ean_Trading_Unit_Identifier=PHMapping.GTIN_CS_Code
'''

postcodes_csv = '/Users/jeanclaudelemoyne/work/Data/Unilever/exogenous/postcodes.csv'
unilever_folder = '/Users/jeanclaudelemoyne/work/Data/Unilever/'
postcode_county_csv = unilever_folder + 'exogenous/postcode_county.csv'
postcode_county_pkl = unilever_folder + 'exogenous/postcode_county.pkl'
income_levels_xlsx = unilever_folder + 'exogenous/NS_Table_3_13_1516.xlsx'
income_csv = unilever_folder + 'exogenous/income_levels.csv'
income_pkl = unilever_folder + 'exogenous/income_levels.pkl'
store_master_csv = unilever_folder + 'store_master.csv'
store_master_income_csv = unilever_folder + 'store_master_income.csv'


def create_postcode_to_county_map(postcode_csvname, postcode_county_csv, postcode_county_pkl):
    df = pd.read_csv(postcode_csvname, low_memory=False)
    pcf = df[['Postcode', 'County']]
    pcf.to_csv(postcode_county_csv, index=False)
    pcdcounty = OrderedDict()
    for index, row in pcf.iterrows():
        if index % 1000 == 0:
            print '... ', index
        pcd = row['Postcode']
        county = row['County']
        if isinstance(county, str):
            pcdcounty[pcd] = county.upper()
    print '# postcode to county: ', len(pcdcounty)
    pickle.dump(pcdcounty, open(postcode_county_pkl, 'wb'))


def readExelFile_to_Income(xcel_fname, income_csv, income_pkl):
    xls = pd.ExcelFile(xcel_fname)
    sheet_names = xls.sheet_names
    print sheet_names
    for book in sheet_names:
        paragraphs = ['United Kingdom', 'England', 'North East ', 'North West ',
                      'Yorkshire and the Humber', 'East Midlands', 'West Midlands',
                      'East of England', 'South East', 'South West ', 'Unitary Authorities']
        print ' ========= ', book, ' ========='
        df = pd.read_excel(xls, sheet_name=book, usecols='A:Z')
        # csv_fname = hh_folder + book + '.csv'
        # df.to_csv(csv_fname, header=True, index=False, encoding='utf-8')
        print df
        print '++++++++++++++++++++++++'
        AtoZ = map(chr, range(ord('A'), ord('Z')+1))
        df.columns = AtoZ
        print df.columns.values
        counties = df['A']
        means = df['T']
        medians = df['U']
        income_stats = OrderedDict()
        for index in range(12, 93):
            tag = counties[index]
            if not isinstance(tag, float) and tag not in paragraphs:
                tag = tag.upper().strip()
                income_stats[tag] = OrderedDict()
                income_stats[tag]['mean'] = means[index]
                income_stats[tag]['median'] = medians[index]
            print index, counties[index], means[index]

        print '~~~~~~~~~~~~~~~~~~~~~'
        print '# of rows: ', len(counties)

        g = open(income_csv, 'w')
        g.write('counti,mean,median\n')
        for tag in income_stats:
            g.write('%s,%s,%s\n' % (tag.strip(), income_stats[tag]['mean'], income_stats[tag]['median']))
            print '%s \tmean: %s  median: %s ' % (tag, income_stats[tag]['mean'], income_stats[tag]['median'])
        g.close()
        pickle.dump(income_stats, open(income_pkl, 'wb'))


def get_store_income(store_master_csv, postcode_county_pkl,
                     income_pkl, store_master_income_csv):
    dfstore = pd.read_csv(store_master_csv, low_memory=False)
    # print dfstore.columns.values
    print 'store master loaded ...'
    dfsp = dfstore[['Retailer_Store_Identifier', 'Location_Postal_Identifier']]
    pcd_county = pickle.load(open(postcode_county_pkl, 'rb'))

    # if 'DY8 1YD' in pcd_county:
    #     print '==================='
    #     return

    print 'postal code to county loaded ...'
    income = pickle.load(open(income_pkl, 'rb'))
    xmean = []
    xmedian = []
    for county in income:
        xmean  =  xmean + [income[county]['mean']]
        xmedian = xmedian + [income[county]['median']]

    amean = np.array(xmean)
    amedian = np.array(xmedian)

    mean_std = np.std(amean)
    median_std = np.std(amedian)


    storeids = OrderedDict()

    # This will eliminate duplicates
    for index, row in dfsp.iterrows():
        if index % 1000 == 0:
            print '... ', index
        stid = row['Retailer_Store_Identifier']
        pcd = row['Location_Postal_Identifier']
        if isinstance(pcd, str):
            pcd = pcd.strip()
            storeids[stid] = pcd

    stinc = OrderedDict()
    mu = 33400
    med = 23200
    nm = 0
    npcd = 0
    nn = 0
    for stid in storeids:
        mean = np.random.normal(mu, mean_std)
        median = np.random.normal(med, median_std)
        nn = nn + 1
        pcd = storeids[stid]
        print pcd
        county = pcd_county['DY8 1YD']
        # print pcd, ' =================== ', county, ' <<<<'

        if pcd in pcd_county:
            county = pcd_county[pcd]
            npcd = npcd + 1
            tokens = county.split(' ')
            for tok in tokens:
                if tok in income:
                    # print '~~~~~>> ', tok
                    mean = income[tok]['mean']
                    median = income[tok]['median']
                    nm = nm + 1
        stinc[stid] = OrderedDict()
        stinc[stid]['mean'] = int(mean)
        stinc[stid]['median'] = int(median)
        # print stid, mean, median
        # if nn > 100:
        #     break

    g = open(store_master_income_csv, 'w')
    g.write('store_id,mean,median\n')
    for stid in stinc:
        g.write('%s,%s,%s\n' % (stid, stinc[stid]['mean'], stinc[stid]['median']))
    g.close()

    print '# postcode matched: ', npcd
    print '# county matched: ', nm

    # print income.keys()
    print 'mean std: ', mean_std
    print 'median std: ', median_std


if __name__ == '__main__':
    print 'Unilever: Google ETL started'
    # create_postcode_to_county_map(postcodes_csv, postcode_county_csv, postcode_county_pkl)
    # readExelFile_to_Income(income_levels_xlsx, income_csv, income_pkl)
    get_store_income(store_master_csv, postcode_county_pkl, income_pkl, store_master_income_csv)

