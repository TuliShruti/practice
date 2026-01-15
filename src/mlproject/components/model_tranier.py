import os
import sys
import numpy as np
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
from src.mlproject.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                "Logistic Regression": LogisticRegression(),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
            }

            params = {
                "Logistic Regression": {},
                "Decision Tree": {"max_depth": [5, 10, 15], "min_samples_split": [2, 5, 10]},
                "Random Forest": {
                    "n_estimators": [100, 500, 1000],
                    "max_depth": [5, 8, 15, None],
                    "min_samples_split": [2, 8, 15],
                    "max_features": [5, 7, "auto"],
                },
                "Gradient Boosting": {
                    "n_estimators": [100, 200, 500],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "max_depth": [3, 5, 7],
                },
            }

            model_report: dict = evaluate_models(
                X_train, y_train, X_test, y_test, models, params
            )

            # Get best model score
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found", sys)

            logging.info(f"Best model found: {best_model_name} with score: {best_model_score}")

            # Train best model with best hyperparameters
            if best_model_name == "Random Forest":
                best_model = RandomForestClassifier(
                    n_estimators=1000,
                    min_samples_split=2,
                    max_features=7,
                    max_depth=None,
                    random_state=42,
                )
            
            best_model.fit(X_train, y_train)

            # Make predictions
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            # Calculate metrics
            train_accuracy = accuracy_score(y_train, y_train_pred)
            test_accuracy = accuracy_score(y_test, y_test_pred)
            train_f1 = f1_score(y_train, y_train_pred, average="weighted")
            test_f1 = f1_score(y_test, y_test_pred, average="weighted")
            test_precision = precision_score(y_test, y_test_pred)
            test_recall = recall_score(y_test, y_test_pred)
            test_roc_auc = roc_auc_score(y_test, y_test_pred)

            logging.info(f"Model Training Completed")
            logging.info(f"Train Accuracy: {train_accuracy}")
            logging.info(f"Test Accuracy: {test_accuracy}")
            logging.info(f"Test F1 Score: {test_f1}")
            logging.info(f"Test ROC AUC Score: {test_roc_auc}")

            # Save the model
            os.makedirs(
                os.path.dirname(self.model_trainer_config.trained_model_file_path),
                exist_ok=True,
            )
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            return {
                "model_name": best_model_name,
                "train_accuracy": train_accuracy,
                "test_accuracy": test_accuracy,
                "train_f1": train_f1,
                "test_f1": test_f1,
                "test_precision": test_precision,
                "test_recall": test_recall,
                "test_roc_auc": test_roc_auc,
            }

        except Exception as e:
            raise CustomException(e, sys)
