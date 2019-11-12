# -*- coding: utf-8 -*-
"""
Created on Mon May 13 11:28:07 2019
@author: sriharis
"""
import pandas as pd
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


class DataFrameUtils:

    def __init__(self):
        self.__labelencoders = {}
        self.__onehotencoders = {}
        self.__scalers = {}

    # ----------------------------------------------------------------------------------------------------------------------
    # ENCODING UTILS
    # ----------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def get_label_encoded(self, df, colname, inplace=True):
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

        le = LabelEncoder()
        le.fit(df[colname])
        le_colname = colname
        if not inplace:
            le_colname = "le_" + le_colname
        df[le_colname] = le.transform(df[colname])
        return df, le

    def labelencode_collist(self, df, collist, inplace=True):
        """
        Returns label encoded columns appended to the data frame.
        New columns are pre-pended with "le_" followed by @colname
        :param df: data frame
        :param collist: list with names of the columns to encode
        :param inplace: if True, replaces the original columns instead of making new ones
        :return: updated dataframe and dict of colname:encoder
        """

        for col in collist:
            if col not in df.columns:
                continue
            df, le = self.get_label_encoded(df, col, inplace)
            #         encoder_list[col] = le
            self.__labelencoders[col] = dict(zip(le.classes_, le.transform(le.classes_)))

        return df

    @staticmethod
    def get_onehot_encoded(self, df, colname, drop_original=True):
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

        ohe = OneHotEncoder(categorical_features=[0], handle_unknown="ignore")
        out = ohe.fit_transform(df[colname].values.reshape(-1, 1)).toarray()
        # Drop the first column - dummy variable trap
        out = out[:, 1:]
        # Join with the original data frame
        dfOneHot = pd.DataFrame(out,
                                columns=[colname + "_" + str(int(i)) for i in range(out.shape[1])],
                                index=df.index)
        df = pd.concat([df, dfOneHot], axis=1)

        if drop_original:
            df.drop(colname, axis=1, inplace=True)

        return df, ohe

    def onehotencode_collist(self, df, collist, drop_original=True):
        """
        Returns One Hot Encoded columns appended to the data frame.
        New columns are pre-pended with @colname followed by encoded class label
        :param df: data frame
        :param collist: list with names of the columns to encode
        :param drop_original: if True, drops original column
        :return: updated dataframe and dict of colname:encoder
        """

        for col in collist:
            if col not in df.columns:
                continue
            print(col)
            df, ohe = self.get_onehot_encoded(df, col, drop_original)
            self.__onehotencoders[col] = ohe

        return df

    # ------------------------------------------------------------------------------------------------------------------
    # SCALING UTILS
    # ------------------------------------------------------------------------------------------------------------------

    def scale_collist(self, df, collist):
        """
        Returns One Hot Encoded columns appended to the data frame.
        New columns are pre-pended with @colname followed by encoded class label
        :param df: data frame
        :param collist: list with names of the columns to encode
        :param drop_original: if True, drops original column
        :return: updated dataframe and dict of colname:encoder
        """

        for col in collist:
            if col not in df.columns:
                continue

            scaler = StandardScaler()
            df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1))
            self.__scalers[col] = scaler

        return df

