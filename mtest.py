import gspread
import pandas as pd
import joblib
import pickle
from oauth2client.service_account import ServiceAccountCredentials
from statsmodels.tsa.stattools import adfuller
from tqdm import tqdm_notebook
import statsmodels.tsa.api as smt
import statsmodels.api as sm
from itertools import product
from statsmodels.tsa.statespace.sarimax import SARIMAXResults

# 
# data = pd.read_csv('https://docs.google.com/spreadsheets/d/1Sy34s0VXEBxLzoAQsCPNZR51pHKAfYwEzM-ZmGZshFY/edit#gid=797493677', error_bad_lines=False)
# scope = ['https://spreadsheets.google.com/feeds',
#                       'https://www.googleapis.com/auth/drive']
# creds = ServiceAccountCredentials.from_json_keyfile_name('./keys.json',scope)
# client = gspread.authorize(creds)
# sheet = client.open('database')
# sheet_instance = sheet.get_worksheet(0)
# records_data = sheet_instance.get_all_records()
# records_df = pd.DataFrame.from_dict(records_data)
# 
# records_df['Hour'] = pd.to_datetime(records_df['Hour'])
# 
# new_set = records_df.set_index("Hour")
# 
# final_set = new_set['Four-Wheeler']
# results = adfuller(final_set)
# print(results[1])


load1 = SARIMAXResults.load('daily_Two-Wheeler_model.pkl')

#predictionss = load1.get_forecast(steps=2)
predictionss = load1.get_prediction(start=pd.to_datetime('2021-09-09'),dynamic=False)
val = predictionss.predicted_mean
row = 63
act = int(max(0,round(val[0])))
print(act)
