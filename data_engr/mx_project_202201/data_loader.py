import pandas as pd 
import numpy as np 
import os

def read_daily_data(daily_d_path):
    date = daily_d_path.split('/')[-1]
    factor_df = pd.read_csv(daily_d_path + '/factor_values.csv', header = None)
    factor_name_list = ['factor_' + str(i) for i in range(factor_df.shape[1])]
    factor_df.columns = factor_name_list

    raw_df = pd.read_csv( daily_d_path + '/ru2005_{date}.csv'.format(date=date))
    times_df = pd.read_csv( daily_d_path + '/times.csv', names = ['time_stamp'])
    y_df = pd.read_csv( daily_d_path + '/y_values.csv', names = ['target'])
    return factor_df, raw_df, times_df, y_df

def read_all_data(raw_data_path):
    dates_list = [x for x in sorted(os.listdir(raw_data_path)) if len(x)==8]
    factor_df_list = []
    y_df_list = []
    times_df_list = []
    for date in dates_list:
        cur_daily_path = raw_data_path + '/' + date
        factor_df, raw_df, times_df, y_df = read_daily_data(cur_daily_path)
        # print(date, factor_df.shape[0], raw_df.shape[0],  times_df.shape[0], y_df.shape[0])
        factor_df_list.append(factor_df)
        y_df_list.append(y_df)
        times_df_list.append(times_df)
    return factor_df_list, y_df_list, times_df_list
