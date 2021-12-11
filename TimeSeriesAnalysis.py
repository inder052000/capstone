
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAXResults
import numpy as np
import gspread
from oauth2client.service_account import ServiceAccountCredentials


# preparing independent and dependent features
class TimeSeriesPrediction:
    def __init__(self):
        self.data = pd.read_csv('https://docs.google.com/spreadsheets/d/1Sy34s0VXEBxLzoAQsCPNZR51pHKAfYwEzM-ZmGZshFY/edit#gid=797493677', error_bad_lines=False)
        # define the scope
        self.scope = ['https://spreadsheets.google.com/feeds',
                'https://www.googleapis.com/auth/drive']
        # add credentials to the account
        self.creds = ServiceAccountCredentials.from_json_keyfile_name('./keys.json', self.scope)
        # authorize the clientsheet
        self.client = gspread.authorize(self.creds)
        # get the instance of the Spreadsheet
        self.sheet = self.client.open('database')
        # get the first sheet of the Spreadsheet
        self.sheet_instance = self.sheet.get_worksheet(2)
        # get all the records of the data
        self.records_data = self.sheet_instance.get_all_records()
        self.records_df = pd.DataFrame.from_dict(self.records_data)
    def updateData(self, data, type_vehicle, time):
        cell_row = 2
        cell_column = 1

        if type_vehicle == 'Two-wheeler' and time == 'daily':
            cell_column = 1
        if type_vehicle == 'Two-wheeler' and time == 'weekly':
            cell_column = 4
        if type_vehicle == 'Two-wheeler' and time == 'monthly':
            cell_column = 7
        if type_vehicle == 'Four-wheeler' and time == 'daily':
            cell_column = 2
        if type_vehicle == 'Four-wheeler' and time == 'weekly':
            cell_column = 5
        if type_vehicle == 'Four-wheeler' and time == 'monthly':
            cell_column = 8
        if type_vehicle == 'Pedestrian' and time == 'daily':
            cell_column = 3
        if type_vehicle == 'Pedestrian' and time == 'weekly':
            cell_column = 6
        if type_vehicle == 'Pedestrian' and time == 'monthly':
            cell_column = 9

        update_sheet_instance = self.sheet.get_worksheet(4)
        # Update Cell
        update_sheet_instance.update_cell(cell_row, cell_column, str(data))

    def calculateAndUploadData(self, field, time,new_date):
        name = f'{time}_{field}_model.pkl'
        load1 = SARIMAXResults.load(name)
        predictionss = load1.get_prediction(start=new_date,dynamic=False)
        val = predictionss.predicted_mean
        act = int(max(0,round(val[0])))
        print(act)
        #calculated_data = val
        self.updateData(act, field, time)
tobj=TimeSeriesPrediction()

tobj.calculateAndUploadData('Two-wheeler', 'daily', '2021-09-08')
tobj.calculateAndUploadData('Two-wheeler', 'weekly', '2022-01-02')
tobj.calculateAndUploadData('Two-wheeler', 'monthly', '2022-02-03')
# tobj.calculateAndUploadData('Four-wheeler', 'Daily', '2021-09-08')
tobj.calculateAndUploadData('Four-wheeler', 'weekly', '2022-01-02')
tobj.calculateAndUploadData('Four-wheeler', 'monthly', '2022-02-03')
tobj.calculateAndUploadData('Pedestrian', 'daily', '2021-09-08')
tobj.calculateAndUploadData('Pedestrian', 'weekly', '2022-01-02')
tobj.calculateAndUploadData('Pedestrian', 'monthly', '2022-02-03')
