import os
import sys
from dataclasses import dataclass
import numpy as np
from sklearn.ensemble import (
    VotingClassifier,
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier
)
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("data_source", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_df, y_train, validation_df, y_test):
        try:
            logging.info("Starting model training with a robust ensemble")

            # Convert target variables to integers
            y_train = y_train.astype(int)
            y_test = y_test.astype(int)

            # Define models with fixed parameters
            xgb_clf = XGBClassifier(
                objective='multi:softmax',
                num_class=len(np.unique(y_train)),
                random_state=42,
                scale_pos_weight=1,
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1
            )

            lgbm_clf = LGBMClassifier(
                class_weight='balanced',
                random_state=42,
                n_estimators=80,
                max_depth=4,
                learning_rate=0.1
            )

            rf_clf = RandomForestClassifier(
                class_weight='balanced',
                random_state=42,
                n_estimators=200,
                max_depth=10
            )

            gb_clf = GradientBoostingClassifier(
                random_state=42,
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6
            )

            et_clf = ExtraTreesClassifier(
                class_weight='balanced',
                random_state=42,
                n_estimators=200,
                max_depth=10
            )

            lr_clf = LogisticRegression(
                class_weight='balanced',
                max_iter=1000,
                random_state=42
            )

            # Voting ensemble with weighted classifiers
            voting_clf = VotingClassifier(
                estimators=[
                    ('xgb', xgb_clf),
                    ('lgbm', lgbm_clf),
                    ('rf', rf_clf),
                    ('gb', gb_clf),
                    ('et', et_clf),
                    ('lr', lr_clf)
                ],
                voting='soft',  # Use probabilities for weighted voting
                weights=[2, 2, 1, 1, 1, 1]  # Adjust weights based on model strength
            )

            logging.info("Fitting the ensemble model")
            voting_clf.fit(train_df, y_train)

            # Evaluate on validation set
            y_pred = voting_clf.predict(validation_df)
            
            # Get detailed metrics
            validation_metrics = self.evaluate_model(y_test, y_pred)
            logging.info(f"Validation Metrics: {validation_metrics}")
            
            # Log detailed classification report
            logging.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")

            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=voting_clf
            )

            return validation_metrics['weighted_f1']  # Return weighted F1 score as final metric

        except Exception as e:
            logging.error("Error in initiate_model_trainer: %s", str(e))
            raise CustomException(e, sys)

    def evaluate_model(self, y_true, y_pred):
        """Evaluate model with multiple metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'macro_f1': f1_score(y_true, y_pred, average='macro'),
            'weighted_f1': f1_score(y_true, y_pred, average='weighted'),
            'macro_precision': precision_score(y_true, y_pred, average='macro'),
            'macro_recall': recall_score(y_true, y_pred, average='macro')
        }
        return metrics
