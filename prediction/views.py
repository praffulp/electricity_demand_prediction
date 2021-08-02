from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.views import APIView
from prediction.apps import PredictionConfig
import pandas as pd
import datetime
import xgboost as xgb
import numpy as np
# Create your views here.
# Class based view to predict based on xgboost model
class Demand_Model_Predict(APIView):
    def post(self, request, format=None):
        data = request.data
        # print(data)
        keys = []
        values = []
        for key in data:
            keys.append(key)
            values.append(data[key])
        dates_list = [datetime.datetime.strptime(date, "%Y-%m-%d") for date in values]
        df = pd.DataFrame(dates_list, columns=['date'], index=dates_list)
        processed = create_features(df)
        # print(keys, values)

        X = processed
        X = xgb.DMatrix(X)
        loaded_mlmodel = PredictionConfig.mlmodel
        y_pred = loaded_mlmodel.predict(X)
        y_pred = pd.Series(y_pred)
        print(y_pred)
        # response_dict = {"Predicted Daily Demand": y_pred[0]}
        zip_response_dict = zip(values, y_pred.values)
        response_dict = dict(zip_response_dict)
        return Response(response_dict, status=200)


def create_features(df, label=None):
    print(df['date'][0])
    print(type(df['date'][0]))
    df['dayofweek'] = df['date'].dt.dayofweek
    df['quarter'] = df['date'].dt.quarter
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['dayofyear'] = df['date'].dt.dayofyear
    df['dayofmonth'] = df['date'].dt.day
    df['weekofyear'] = df['date'].dt.weekofyear
    df['demand_7_days_mean'] = 0
    df['demand_15_days_mean'] = 0
    df['demand_30_days_mean'] = 0
    df['demand_7_days_std'] = 0
    df['demand_15_days_std'] = 0
    df['demand_30_days_std'] = 0
    df['demand_7_days_max'] = 0
    df['demand_15_days_max'] = 0
    df['demand_30_days_max'] = 0
    df['demand_7_days_min'] = 0
    df['demand_15_days_min'] = 0
    df['demand_30_days_min'] = 0

    columns = ['dayofweek','quarter','month','year','dayofyear','dayofmonth',
               'weekofyear']

    for no_of_days in ('7', '15', '30'):
        for measure_type in ('mean', 'std', 'max', 'min'):
            columns.append(f'demand_{no_of_days}_days_{measure_type}')

    X = df[columns]
    if label:
        y = df[label]
        return X, y
    return X