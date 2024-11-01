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
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from scipy.stats import randint, uniform

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("data_source", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_df, y_train, validation_df, y_test):
        try:
            logging.info("Starting model training with a robust ensemble and hyperparameter tuning")

            # Convert target variables to integers
            y_train = y_train.astype(int)
            y_test = y_test.astype(int)

            # Define models with initial parameters
            xgb_clf = XGBClassifier(
                objective='multi:softmax',
                num_class=len(np.unique(y_train)),
                random_state=42,
                scale_pos_weight=1
            )

            lgbm_clf = LGBMClassifier(
                class_weight='balanced',
                random_state=42
            )

            rf_clf = RandomForestClassifier(
                class_weight='balanced',
                random_state=42
            )

            gb_clf = GradientBoostingClassifier(
                random_state=42
            )

            et_clf = ExtraTreesClassifier(
                class_weight='balanced',
                random_state=42
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

            # Parameter grid for RandomizedSearch
            param_distributions = {
                'xgb__n_estimators': randint(100, 300),
                'xgb__max_depth': randint(3, 10),
                'xgb__learning_rate': uniform(0.05, 0.3),
                'lgbm__n_estimators': randint(100, 300),
                'lgbm__max_depth': randint(6, 15),
                'lgbm__learning_rate': uniform(0.05, 0.3),
                'lgbm__min_child_samples': randint(10, 30),
                'lgbm__min_split_gain': uniform(0.0, 0.5),
                'lgbm__lambda_l1': uniform(0.0, 1.0),
                'lgbm__lambda_l2': uniform(0.0, 1.0),
                'rf__n_estimators': randint(100, 300),
                'rf__max_depth': randint(3, 10),
                'gb__n_estimators': randint(100, 300),
                'gb__learning_rate': uniform(0.05, 0.3),
                'gb__max_depth': randint(3, 10),
                'et__n_estimators': randint(100, 300),
                'et__max_depth': randint(3, 10)
            }

            # StratifiedKFold for handling class imbalance in cross-validation
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

            # Perform randomized search
            random_search = RandomizedSearchCV(
                voting_clf,
                param_distributions=param_distributions,
                n_iter=50,  # Adjust based on computational resources
                cv=cv,
                scoring='f1_macro',
                refit=True,
                n_jobs=-1,
                verbose=1
            )

            logging.info("Starting randomized search with cross-validation")
            random_search.fit(train_df, y_train)

            # Get best parameters and scores
            best_params = random_search.best_params_
            best_score = random_search.best_score_
            
            logging.info(f"Best Parameters: {best_params}")
            logging.info(f"Best cross-validation F1 Macro score: {best_score}")

            # Evaluate on validation set
            best_model = random_search.best_estimator_
            y_pred = best_model.predict(validation_df)
            
            # Get detailed metrics
            validation_metrics = self.evaluate_model(y_test, y_pred)
            logging.info(f"Validation Metrics: {validation_metrics}")
            
            # Log detailed classification report
            logging.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")

            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
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
