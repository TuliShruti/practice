from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
from src.mlproject.components.data_ingestion import DataIngestion
from src.mlproject.components.data_transformation import DataTransformation
from src.mlproject.components.model_trainer import ModelTrainer  # FIXED

import sys


if __name__ == "__main__":
    logging.info("The execution has started")

    try:
        # Data Ingestion
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

        # Data Transformation
        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
            train_data_path, test_data_path
        )

        # Model Training
        model_trainer = ModelTrainer()
        result = model_trainer.initiate_model_trainer(train_arr, test_arr)

        logging.info(f"Model Trainer Result: {result}")
        print("\n===== MODEL TRAINING RESULT =====\n")
        print(result)

    except Exception as e:
        logging.info("Custom Exception")
        raise CustomException(e, sys)
