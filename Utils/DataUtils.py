"""
@file_name : DataUtils.py
@author : Srihari Seshadri
@description : This file contains methods for data wrangling and other array/set operations
@date : 01-29-2019
"""

import pandas as pd
import numpy as np
import math


def get_ranks(array):
    temp = array.argsort()
    ranks = np.empty_like(temp)
    ranks[temp] = np.arange(len(array))
    return ranks

def num_nan_rows(df):
    return df.shape[0] - df.dropna().shape[0]


def find_common_elems(list_of_lists):
    """
    Find common elems in all the lists given in the list of lists using sets (intersection)
    :param list_of_lists: list of lists of elements
    :return: list of common elements
    """
    common_elems = list_of_lists[0]
    for i in range(1, len(list_of_lists)):
        common_elems = list(set(common_elems).intersection(list_of_lists[i]))
    return common_elems


def find_unique_elems(list_of_lists):
    """
    Find unique non overlapping elems in all the lists given in the list of lists using sets
    :param list_of_lists: list of lists of elements
    :return: list of unique elements
    """
    unique_elems = set(list_of_lists[0])
    for i in range(1, len(list_of_lists)):
        unique_elems = set(list_of_lists[i]) ^ unique_elems
    return list(unique_elems)


def analyse_nans(df):
    """
    Returns a dataframe with details on how many NaNs and where from a given dataframe
    :param df: dataframe
    :return: NaN dataframe
    """
    temp_df = pd.DataFrame(columns=df.columns,
                           index=["total", "percentage", "idx_list"])
    for col in df.columns:
        idxes = df[col].isnull()
        num_nans = idxes.sum(axis = 0)
        nan_pct = 100*np.round(num_nans/df.shape[0], 3)
        temp_df[col] = [num_nans, nan_pct,
                        df.index[idxes.values == True].tolist()]
    return temp_df


def describe_unique(df, colname, filter_unnecessary=True):
    """
    Describes all the unique elements in a column
    :param df: Dataframe
    :param colname: Column name
    :param filter_unnecessary: IF True, IF all values are unique, returns nothing
    :return: prints details on all the unique elements in column
    """
    print("Column name : ", colname)
    unique_elems = pd.unique(df[colname])
    types_of_data = [type(x) for x in unique_elems]
    if filter_unnecessary:
        if len(unique_elems) == df.shape[0]:
            print("All values are unique.")
            return
    print("Number of unique elems : ", len(unique_elems))
    print("Types of data in col :", set(types_of_data))
    for idx, uel in zip(range(0, len(unique_elems)), unique_elems):
        print("  ", str(idx)+".", type(uel), "\t",uel)


def merge_replace(left, right, left_on, right_on, how, drop_list):
    """
    Merges 2 datasets and drops the columns specified in the list
    :param left: left dataframe
    :param right: right dataframe
    :param left_on: left_on key
    :param right_on: right_on key
    :param how: "inner", "outer", "left", "right"
    :param drop_list: list of cols to drop
    :return: merged dataframe
    """
    left = pd.merge(left=left, right=right, left_on=left_on, right_on=right_on, how=how)
    left.drop(drop_list, axis=1, inplace=True)
    return left


def cartesian_product(left, right):
    """
    Performs cartesian product of 2 dataframes
    :param left: left dataframe
    :param right: right dataframe
    :return: cartesian product of leftxright dataframe
    """
    la, lb = len(left), len(right)
    ia2, ib2 = np.broadcast_arrays(*np.ogrid[:la,:lb])

    return pd.DataFrame(
        np.column_stack([left.values[ia2.ravel()], right.values[ib2.ravel()]]))