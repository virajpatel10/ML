import os
import sys

from sklearn.model_selection import train_test_split
import pandas as pd

from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation,DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig, ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts","train.csv")
    test_data_path: str = os.path.join("artifacts","test.csv")
    raw_data_path: str = os.path.join("artifacts","data.csv")


class DataIngestion:

    def __init__(self):
        self.data_ingestion_config=DataIngestionConfig()

    def initial_data_ingestion(self):
        logging.info("Enter the data ingestion")
        try:
            df=pd.read_csv('notebook/data/stud.csv')
            logging.info('raw Data ingestion done')

            os.makedirs(os.path.dirname(self.data_ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.data_ingestion_config.raw_data_path,index=False,header=True)

            logging.info("split initiated")

            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)

            train_set.to_csv(self.data_ingestion_config.train_data_path,index=False,header=True)

            test_set.to_csv(self.data_ingestion_config.test_data_path,index=False,header=True)

            logging.info("data ingestion done")

            return (
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path

            )
        
        except Exception as e:
            raise CustomException(e,sys)



if __name__ == "__main__":
    obj = DataIngestion()
    train_data,test_data=obj.initial_data_ingestion()

    data_transformation=DataTransformation()
    train_arr, test_arr,_= data_transformation.initiate_data_transformation(train_data,test_data)

    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))