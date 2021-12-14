import gspread
import pandas as pd
from oauth2client.service_account import ServiceAccountCredentials
import numpy as np
import pickle
from statsmodels.tsa.stattools import adfuller
from tqdm import notebook
import statsmodels.tsa.api as smt
import statsmodels.api as sm
from itertools import product
import joblib

sheet_no=[0,1,2]
vehicle_type=['Two-wheeler','Four-wheeler','Pedestrian']
tfield=['daily','weekly','monthly']
data = pd.read_csv('https://docs.google.com/spreadsheets/d/1Sy34s0VXEBxLzoAQsCPNZR51pHKAfYwEzM-ZmGZshFY/edit#gid=797493677', error_bad_lines=False)
scope = ['https://spreadsheets.google.com/feeds',
                      'https://www.googleapis.com/auth/drive']
creds = ServiceAccountCredentials.from_json_keyfile_name('./keys.json',scope)
client = gspread.authorize(creds)
sheet = client.open('database')

def optimize_SARIMA(parameters_list, d, D, s,data):
    """
        Return dataframe with parameters and corresponding AIC
        
        parameters_list - list with (p, q, P, Q) tuples
        d - integration order
        D - seasonal integration order
        s - length of season
    """
    results = []
    best_aic = float('inf')
    k=1
    for param in notebook.tqdm(parameters_list):
        print(f'---------------------------------------------{k}--------------')
        k+=1
        try: model = sm.tsa.statespace.SARIMAX(data, order=(param[0], d, param[1]),
                                               seasonal_order=(param[2], D, param[3], s)).fit()
        except Exception as e:
            continue
            
        aic = model.aic
        
        #Save best model, AIC and parameters
        if aic < best_aic:
            best_model = model
            best_aic = aic
            best_param = param
        results.append([param, model.aic])
        
    result_table = pd.DataFrame(results)
    result_table.columns = ['parameters', 'aic']
    #Sort in ascending order, lower AIC is better
    result_table = result_table.sort_values(by='aic', ascending=True).reset_index(drop=True)
    return result_table

def create_models():
    for stt in sheet_no:
        for vt in vehicle_type:
            sheet_instance = sheet.get_worksheet(stt)
            records_data = sheet_instance.get_all_records()
            records_df = pd.DataFrame.from_dict(records_data)
            n_steps = 5
            n_features = 1
            print('After n_features')
            records_df['Hour'] = pd.to_datetime(records_df['Hour'])

            new_set = records_df.set_index("Hour")
            final_set = new_set[vt]
            results = adfuller(final_set)
            print(results[1])


            valued=2
            ps = range(0, valued)
            d = 1
            qs = range(0, valued)
            Ps = range(0, valued)
            D = 1
            Qs = range(0, valued)
            s = 5
    #Create a list with all possible combinations of parameters
            parameters = product(ps, qs, Ps, Qs)
            parameters_list = list(parameters)
            print("Before result_table")
            # Train many SARIMA models to find the best set of parameters
            result_table = optimize_SARIMA(parameters_list, d, D, s,final_set)
            print("After result_table")
            #Set parameters that give the lowest AIC (Akaike Information Criteria)
            p, q, P, Q = result_table.parameters[0]
            print("Before best_model")
            best_model = sm.tsa.statespace.SARIMAX(final_set, order=(p, d, q),
                                                seasonal_order=(P, D, Q, s)).fit()

            print("After result_table")
            best_model.save(f'./models/{tfield[stt]}_{vt}_model.pkl')
            print(f'./models/{tfield[stt]}_{vt}_model.pkl created------------------')