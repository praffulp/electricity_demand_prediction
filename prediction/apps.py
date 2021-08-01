from django.apps import AppConfig
import xgboost as xgb
import joblib
import os

class PredictionConfig(AppConfig):
    name = 'prediction'
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    MLMODEL_FOLDER = os.path.join(BASE_DIR+'/ml_models/')
    print("here", MLMODEL_FOLDER)
    MLMODEL_FILE = os.path.join(MLMODEL_FOLDER, 'model_xgb.txt')
    mlmodel = xgb.Booster()
    mlmodel.load_model(MLMODEL_FILE)