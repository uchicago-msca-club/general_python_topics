# -*- coding: utf-8 -*-
"""
Created on Mon May 13 11:28:07 2019
@author: sriharis
"""
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd 
import numpy as np
import os
import pickle

# Main

# Multiple categories
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
# Definition of the CategoricalEncoder class, copied from PR #9151.
# Just run this cell, or copy it to your code, do not try to understand it (yet).

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.preprocessing import LabelEncoder
from scipy import sparse
from sklearn.pipeline import FeatureUnion


# ----------------------------------------------------------------------------------------------------------------------
# FILE UTILS
# ----------------------------------------------------------------------------------------------------------------------

def save_to_disk(obj, filename):
    try:
        with open(filename, 'wb') as handle:
            pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print(e)


def load_from_disk(filename):
    try:
        with open(filename, 'rb') as handle:
            b = pickle.load(handle)        
            return b
    except Exception as e:
        print(e)    


# ----------------------------------------------------------------------------------------------------------------------
# ENCODING UTILS
# ----------------------------------------------------------------------------------------------------------------------

def get_label_encoded(df, colname, inplace=True):
    """
    Returns label encoded column appended to the data frame.
    New column is pre-pended with "le_" followed by @colname
    :param df: data frame
    :param colname: name of the column to encode
    :param inplace: if True, replaces the original columns instead of making new ones
    :return: updated dataframe
    """
    
    # Sanity check
    if colname not in df.columns:
        raise ValueError("Column not in Dataframe!")
        return data
    
    le = LabelEncoder()
    le.fit(df[colname])
    le_colname = colname
    if not inplace:
        le_colname = "le_" + le_colname
    df[le_colname] = le.transform(df[colname])
    return df, le


def labelencode_collist(df, collist, inplace=True):
    """
    Returns label encoded columns appended to the data frame.
    New columns are pre-pended with "le_" followed by @colname
    :param df: data frame
    :param collist: list with names of the columns to encode
    :param inplace: if True, replaces the original columns instead of making new ones
    :return: updated dataframe and dict of colname:encoder
    """
    
    encoder_list= {}
    
    for col in collist:
        if col not in df.columns:
            continue
        df, le = get_label_encoded(df, col, inplace)
#         encoder_list[col] = le
        encoder_list[col] = dict(zip(le.classes_, le.transform(le.classes_)))
        
    return df, encoder_list


def get_onehot_encoded(df, colname, drop_original=True):
    """
    Returns One Hot Encoded columns appended to the data frame.
    New columns are pre-pended with @colname followed by encoded class label
    :param df: data frame
    :param colname: name of the column to encode
    :param drop_original: if True, drops original column
    :return: updated dataframe 
    """
    
    # Sanity check
    if colname not in df.columns:
        raise ValueError("Column not in Dataframe!")
        return data
    
    ohe = OneHotEncoder(categorical_features=[0], handle_unknown="ignore")
    out = ohe.fit_transform(df[colname].values.reshape(-1,1)).toarray()
    # Drop the first column - dummy variable trap
    out = out[:,1:]
    # Join with the original data frame
    dfOneHot = pd.DataFrame(out, 
                            columns=[colname+"_"+str(int(i)) for i in range(out.shape[1])], 
                            index=df.index)
    df = pd.concat([df, dfOneHot], axis=1)
    
    if drop_original:
        df.drop(colname, axis=1, inplace=True)
    
    return df, ohe


def onehotencode_collist(df, collist, drop_original=True):
    """
    Returns One Hot Encoded columns appended to the data frame.
    New columns are pre-pended with @colname followed by encoded class label
    :param df: data frame
    :param collist: list with names of the columns to encode
    :param drop_original: if True, drops original column
    :return: updated dataframe and dict of colname:encoder
    """
    
    encoder_list= {}
    
    for col in collist:
        if col not in df.columns:
            continue
        print(col)
        df, ohe = get_onehot_encoded(df, col, drop_original)
        encoder_list[col] = ohe
        
    return df, encoder_list



# ----------------------------------------------------------------------------------------------------------------------
# SCALING UTILS
# ----------------------------------------------------------------------------------------------------------------------

def scale_collist(df, collist):
    """
    Returns One Hot Encoded columns appended to the data frame.
    New columns are pre-pended with @colname followed by encoded class label
    :param df: data frame
    :param collist: list with names of the columns to encode
    :param drop_original: if True, drops original column
    :return: updated dataframe and dict of colname:encoder
    """
    
    scaler_list = {}
    
    for col in collist:
        if col not in df.columns:
            continue
        
        scaler = StandardScaler()
        df[col] = scaler.fit_transform(df[col].values.reshape(-1,1))
        scaler_list[col] = scaler
        
    return df, scaler_list