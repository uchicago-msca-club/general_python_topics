"""
@author: Srihari Seshadri
@desc : ETL script for populating a MySQL database for the KDD2014 dataset
@date : 12-23-2018
"""

import os
import datetime
import calendar

from Utils.DataExtractor import DataExtractor
from Utils.UtilsViz import *


# Week to start on Sunday [0-6] where 0 is monday and 6 is sunday
calendar.setfirstweekday(6)


def get_week_of_month(year, month, day):
    x = np.array(calendar.monthcalendar(year, month))
    week_of_month = np.where(x == day)[0][0] + 1
    return week_of_month


def add_date_info(data, date_col_name):
    # Convert the Date column into a datetime type
    data[date_col_name] = pd.to_datetime(data[date_col_name])

    winter_months = [1, 2, 3]
    summer_months = [4, 5, 6]
    spring_months = [7, 8, 9]
    fall_months = [10, 11, 12]
    season_months = {}
    for i in range(13):
        if i in winter_months:
            season_months[i] = 0
        if i in summer_months:
            season_months[i] = 1
        if i in spring_months:
            season_months[i] = 2
        if i in fall_months:
            season_months[i] = 3

    # Plot the aggregated crime count wrt every month
    data["year"] = \
        data.apply(lambda row: row[date_col_name].year, axis=1)
    data["month"] = \
        data.apply(lambda row: row[date_col_name].month, axis=1)
    data["day"] = \
        data.apply(lambda row: row[date_col_name].day, axis=1)
    data["quarter"] = \
        data.apply(lambda row: season_months[row['month']], axis=1)
    data["week_no"] = \
        data.apply(lambda row:
                   datetime.date(row[date_col_name].year,
                                 row[date_col_name].month,
                                 row[date_col_name].day).isocalendar()[1],
                   axis=1)
    return data


# ------------------------------------------------------------------------------------------ #
# Functions to transform data
# ------------------------------------------------------------------------------------------ #


def process_projects_data(df):
    df = add_date_info(df, "date_posted")
    return df


def process_donations_data(df):
    print_df_cols(df)
    # Check and drop for duplicates
    print("Number of duplicate IDs ", df["donationid"].duplicated().sum())
    # Drop them
    df.drop_duplicates(subset=None, keep='first', inplace=True)
    print("Adding date info ")
    df = add_date_info(df, "donation_timestamp")

    return df


def process_resources_data(df):
    print_df_cols(df)
    # Check and drop for duplicates
    print("Number of duplicate IDs ", df["resourceid"].duplicated().sum())

    # Drop them
    df.drop_duplicates(subset=None, keep='first', inplace=True)

    return df


# ------------------------------------------------------------------------------------------ #
# Functions to load data and store to DB
# ------------------------------------------------------------------------------------------ #


def projects_data(file_path):

    data_extractor = DataExtractor()
    data = data_extractor.read_csv(fpath=file_path)
    if data.empty:
        return

    # Clean and transform the dataframe
    # Note: too many NaNs if we check row count to just remove.
    # Do not remove NaNs until further instruction is received.
    data = process_projects_data(data)

    database = 'kdd_2014'
    data_extractor.write_to_db(db=database, df=data, table_name="projects", user="root", pwd="root")


def donations_data(file_path):

    data_extractor = DataExtractor()
    data = data_extractor.read_csv(fpath=file_path)

    # Ugly hack for now due to encoding issues in local MySQL instance (or PyMySQL?) (raise bug)
    data["donation_message"] = ""

    if data.empty:
        return

    # Clean and transform the dataframe
    data = process_donations_data(data)

    database = 'kdd_2014'

    # Split the data into years and append each year to the table
    years = data["year"].unique()
    print(years)
    for year in years:
        print(year)
        year_data = data[data["year"] == year]
        fpath = os.path.join(os.path.dirname(file_path), str(year)+".csv")
        print(fpath)
        data_extractor.write_to_csv(df=year_data, filepath=fpath)
        data_extractor.write_to_db(db=database, df=year_data, table_name="donations",
                                   user="root", pwd="root", if_table_exists="append")

    # data.drop(["Unnamed: 0"], axis=1, inplace=True)
    # data["donation_message"] = data["donation_message"].str.encode('utf-8')
    # Ugly hack for now
    # data["donation_message"] = ""
    # print_df_cols(data)

    # data_extractor.write_to_db(db=database, df=data, table_name="donations",
    #                            user="root", pwd="root", if_table_exists="append")

    del data


def resources_data(file_path):

    data_extractor = DataExtractor()
    data = data_extractor.read_csv(fpath=file_path)
    if data.empty:
        return

    # Clean and transform the dataframe
    data = process_resources_data(data)

    database = 'kdd_2014'
    data_extractor.write_to_db(db=database, df=data, table_name="resources",
                               user="root", pwd="root",
                               if_table_exists="replace")


def main():
    # What are the data files provided?
    data_path = os.path.join(os.getcwd(), "datasets")

    # Projects data
    file_path = os.path.join(data_path, "projects.csv")
    projects_data(file_path)

    # Donations data
    file_path = os.path.join(data_path, "donations.csv")
    # file_path = os.path.join(data_path, "2013.csv")
    donations_data(file_path)

    # Resources data
    file_path = os.path.join(data_path, "resources.csv")
    resources_data(file_path)


if __name__ == "__main__":
    main()
