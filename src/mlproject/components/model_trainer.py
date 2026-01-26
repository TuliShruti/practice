import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    ConfusionMatrixDisplay
)
import mlflow
import mlflow.sklearn
import dagshub

from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
from src.mlproject.utils import save_object, evaluate_models


# DagsHub Configuration
dagshub.init(repo_owner='shrutiguha2002', repo_name='practice', mlflow=True)


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            # Set MLflow experiment
            mlflow.set_experiment("Model Training Pipeline")

            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Decision Tree": DecisionTreeClassifier(),
                "Random Forest": RandomForestClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
            }

            params = {
                "Logistic Regression": {},
                "Decision Tree": {
                    "max_depth": [5, 10, 15],
                    "min_samples_split": [2, 5, 10]
                },
                "Random Forest": {
                    "n_estimators": [100, 500, 1000],
                    "max_depth": [5, 8, 15, None],
                    "min_samples_split": [2, 8, 15],
                    "max_features": [5, 7, "sqrt"],
                },
                "Gradient Boosting": {
                    "n_estimators": [100, 200, 500],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "max_depth": [3, 5, 7],
                },
            }

            # ---------------- MLflow Run 1: Model Evaluation ----------------
            with mlflow.start_run(run_name="Model Evaluation"):
                logging.info("Starting model evaluation with multiple algorithms")

                # IMPORTANT FIX: get trained best model
                model_report, best_model, best_model_name = evaluate_models(
                    X_train, y_train, X_test, y_test, models, params
                )

                best_model_score = max(model_report.values())

                if best_model_score < 0.6:
                    raise CustomException("No best model found", sys)

                logging.info(
                    f"Best model found: {best_model_name} with score: {best_model_score}"
                )

                # Log model comparison metrics
                for model_name, score in model_report.items():
                    mlflow.log_metric(f"model_accuracy_{model_name}", score)

                mlflow.log_param("best_model", best_model_name)

            # ---------------- MLflow Run 2: Final Model ----------------
            with mlflow.start_run(run_name=f"Best Model - {best_model_name}"):

                # best_model is ALREADY TRAINED
                y_train_pred = best_model.predict(X_train)
                y_test_pred = best_model.predict(X_test)

                # Calculate metrics
                train_accuracy = accuracy_score(y_train, y_train_pred)
                test_accuracy = accuracy_score(y_test, y_test_pred)
                train_f1 = f1_score(y_train, y_train_pred, average="weighted")
                test_f1 = f1_score(y_test, y_test_pred, average="weighted")
                test_precision = precision_score(y_test, y_test_pred, zero_division=0)
                test_recall = recall_score(y_test, y_test_pred, zero_division=0)
                test_roc_auc = roc_auc_score(y_test, y_test_pred)

                logging.info("Model Training Completed")
                logging.info(f"Train Accuracy: {train_accuracy}")
                logging.info(f"Test Accuracy: {test_accuracy}")
                logging.info(f"Test F1 Score: {test_f1}")
                logging.info(f"Test ROC AUC Score: {test_roc_auc}")

                # Log metrics
                mlflow.log_metrics({
                    "train_accuracy": train_accuracy,
                    "test_accuracy": test_accuracy,
                    "train_f1": train_f1,
                    "test_f1": test_f1,
                    "test_precision": test_precision,
                    "test_recall": test_recall,
                    "test_roc_auc": test_roc_auc,
                })

                # Log model (CORRECT MODEL)
                mlflow.sklearn.log_model(best_model, "model")

                # Confusion matrix
                cm = confusion_matrix(y_test, y_test_pred)
                fig, ax = plt.subplots(figsize=(8, 6))
                disp = ConfusionMatrixDisplay(confusion_matrix=cm)
                disp.plot(ax=ax)
                plt.title(f"Confusion Matrix - {best_model_name}")
                plt.savefig("confusion_matrix.png")
                mlflow.log_artifact("confusion_matrix.png")
                plt.close()

                # ROC curve
                try:
                    fpr, tpr, _ = roc_curve(
                        y_test, best_model.predict_proba(X_test)[:, 1]
                    )
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.plot(fpr, tpr, label=f'ROC (AUC = {test_roc_auc:.2f})')
                    ax.plot([0, 1], [0, 1], 'r--')
                    ax.set_xlabel('False Positive Rate')
                    ax.set_ylabel('True Positive Rate')
                    ax.set_title(f'ROC Curve - {best_model_name}')
                    ax.legend()
                    plt.savefig("roc_curve.png")
                    mlflow.log_artifact("roc_curve.png")
                    plt.close()
                except Exception as roc_error:
                    logging.info(f"Could not generate ROC curve: {roc_error}")

                # Save locally
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
                    "model_file_path": self.model_trainer_config.trained_model_file_path,
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
