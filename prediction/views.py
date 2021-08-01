from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.views import APIView
from prediction.apps import PredictionConfig
import pandas as pd

# Create your views here.
# Class based view to predict based on xgboost model
class Demand_Model_Predict(APIView):
    def post(self, request, format=None):
        data = request.data
        print(data)
        # keys = []
        # values = []
        # for key in data:
        #     keys.append(key)
        #     values.append(data[key])
        # X = pd.Series(values).to_numpy().reshape(1, -1)
        # loaded_mlmodel = PredictionConfig.mlmodel
        # y_pred = loaded_mlmodel.predict(X)
        # y_pred = pd.Series(y_pred)
        # target_map = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
        # y_pred = y_pred.map(target_map).to_numpy()
        # response_dict = {"Predicted Daily Demand": y_pred[0]}
        response_dict = {"Predicted Daily Demand": [1]}
        return Response(response_dict, status=200)