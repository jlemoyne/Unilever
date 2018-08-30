#!/usr/bin/env python

import os
import pandas as pd
import pickle
import csv

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


def read_xcel(xcel_fname, pkl_fname):
    ss = pd.read_excel(xcel_fname)
    pickle.dump(ss, open(pkl_fname, 'wb'))
    # print ss


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


if __name__ == '__main__':
    if os.path.exists(data_source_2017_2018):
        print '>>> Data from ' + data_source_2017_2018
        if not os.path.exists(unilever_training_folder):
            os.makedirs(unilever_training_folder)

        # read_xcel(data_source_2017_2018, train_source_2017_2018)
        # ssdf_2017_2018 = load_source(train_source_2017_2018)
        # ssdf_2017_2018.to_csv(train_csv_2017_2018, sep='\t', index=False)
        # weka_csv_compliance(train_csv_2017_2018, train_weka_csv_2017_2018)
        check_for_dups()

        # read_xcel(data_source_2016, train_source_2016)
        # ssdf_2016 = load_source(train_source_2016)

        # read_xcel(data_source_2015, train_source_2015)
        # ssdf_2015 = load_source(train_source_2015)
