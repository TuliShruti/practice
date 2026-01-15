
import os
import sys
import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score, accuracy_score
from sklearn.model_selection import GridSearchCV

from src.mlproject.exception import CustomException
from src.mlproject.logger import logging

def save_object(file_path, obj):
    """
    Save the object to a file using dill.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file:
            dill.dump(obj, file)
        print(f"Object saved to {file_path}")
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    """
    Evaluate the models using GridSearchCV and return a dictionary with model names and their best test accuracy scores.
    """
    try:
        model_report = {}
        for name, model in models.items():
            param_grid = params.get(name, {})
            if param_grid:
                gs = GridSearchCV(model, param_grid, cv=3, n_jobs=-1, scoring='accuracy')
                gs.fit(X_train, y_train)
                best_model = gs.best_estimator_
            else:
                best_model = model
                best_model.fit(X_train, y_train)
            y_test_pred = best_model.predict(X_test)
            test_model_score = accuracy_score(y_test, y_test_pred)
            model_report[name] = test_model_score
        return model_report
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    """
    Load an object from a file using dill.
    """
    try:
        with open(file_path, 'rb') as file:
            return dill.load(file)
    except Exception as e:
        raise CustomException(e, sys)