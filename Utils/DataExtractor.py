"""
@file_name : DataExtractor.py
@author : Srihari Seshadri
@description : This file does the following operations :
                1. Reads the datafile.csv for getting all info required
                2. Populates a pandas dataframe
            Potential upgrades include
                1. Analysis modules
@date : 11-14-2018
"""

import pandas as pd
import pickle
from Utils.SQLDatabaseManager import SQLDatabaseManager


class DataExtractor:

    def __init__(self):
        pass

    # -------------------------------------------------------------------------
    # CSV FILES
    # -------------------------------------------------------------------------
    @staticmethod
    def read_csv(fpath, sep=",", names=None, nrows_to_read=None,
                 na_vals=None, **kwargs):
        """
        Populates a pandas dataframe by reading a csv file or buffer
        :param fpath: file_path of .csv or buffer
        :param sep: Delimiter
        :param names: Column names to use when reading csv
        :param nrows_to_read: number of rows to read from file/buffer
        :param na_vals: list of values in cells that will be considered as NA
        :return: dataframe with data if success. Empty dataframe if failure
        """
        try:
            if nrows_to_read is not None:
                if nrows_to_read < 0:
                    nrows_to_read = None
            df = pd.read_csv(filepath_or_buffer=fpath,
                             sep=sep, names=names,
                             nrows=nrows_to_read,
                             na_values=na_vals,
                             **kwargs)
        except Exception as e:
            print(" Exception thrown while reading CSV file/buffer :", e)
            return pd.DataFrame()
        return df


    @staticmethod
    def write_to_csv(self, df, filepath, seperator=","):
        try:
            df.to_csv(filepath, sep=seperator)
        except Exception as e:
            print(" Exception thrown while writing CSV file/buffer :", e)
            return -1
        return 1

    # -------------------------------------------------------------------------
    # DATABASE OPERATIONS
    # -------------------------------------------------------------------------
    @staticmethod
    def read_db(self, db, query, host="localhost", user="root", pwd="root" ):
        # Load the data from the database
        sqldbm = SQLDatabaseManager()

        port = '3306'
        ret = sqldbm.connect(host=host,
                             database=db,
                             username=user,
                             password=pwd,
                             port=port)

        if ret != 1:
            print(" Closing program ")
            exit(-10)

        # Work on this data year by
        data = sqldbm.execute_query(query=query)

        sqldbm.disconnect()

        return data

    @staticmethod
    def write_to_db(self, db, df, table_name,
                    host="localhost", user="root", pwd="Irahirs1!", if_table_exists="replace"):
        # Save the data to the DB
        sqldbm = SQLDatabaseManager()

        port = '3306'

        ret = sqldbm.connect(host=host,
                             database=db,
                             username=user,
                             password=pwd,
                             port=port,
                             charset='utf8mb4')

        if ret != 1:
            print(" Closing program ")
            return

        print(sqldbm.get_tables())

        ret = sqldbm.insert(dframe=df, table_name=table_name, if_table_exists=if_table_exists)

        sqldbm.disconnect()

        return ret

    # -------------------------------------------------------------------------
    # GENERIC OBJECT FILES
    # -------------------------------------------------------------------------
    @staticmethod
    def save_obj_to_disk(self, obj, filename):
        try:
            with open(filename, 'wb') as handle:
                pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print(e)

    @staticmethod
    def load_obj_from_disk(self, filename):
        try:
            with open(filename, 'rb') as handle:
                b = pickle.load(handle)
                return b
        except Exception as e:
            print(e)


def main():

    file_path = ""

    data_extractor = DataExtractor()
    data = data_extractor.read_csv(fpath=file_path)
    if data.empty:
        return
    print(data.head())


if __name__ == "__main__":
    main()
