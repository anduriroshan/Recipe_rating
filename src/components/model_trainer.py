import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import lightgbm as lgb
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("data_source", "stacking_model.pkl")

class StackingModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_stacking_model_trainer(self, X_train, X_test):
        try:
            logging.info("Initializing base classifiers")
            logging.info("Split training and test input data")
            
            X_train,y_train,X_test,y_test=(
                X_train.drop(['Crime_Category'],axis=1),
                X_train.Crime_Category,
                X_test.drop(['Crime_Category'],axis=1),
                X_test.Crime_Category,
            )
            
            # Define base classifiers
            xgb_clf = xgb.XGBClassifier(objective='multi:softmax', num_class=6, learning_rate=0.01)
            lgbm_clf = lgb.LGBMClassifier(learning_rate=0.1, verbose=-1)
            dt_clf = RandomForestClassifier()

            # Define the Stacking Classifier with the base classifiers
            estimators = [('xgb', xgb_clf), ('lgbm', lgbm_clf), ('dt', dt_clf)]
            meta_clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())

            # Define parameter grid for hyperparameter tuning
            param_grid = {
                'xgb__n_estimators': [150],
                'xgb__max_depth': [7],
                'lgbm__n_estimators': [350],
                'lgbm__num_leaves': [35],
                'dt__max_depth': [7]
            }

            # Perform grid search with cross-validation
            logging.info("Starting grid search for hyperparameter tuning")
            grid_search = GridSearchCV(meta_clf, param_grid, cv=5, scoring='accuracy')
            grid_search.fit(X_train, y_train)

            # Get the best parameters and best score
            best_params = grid_search.best_params_
            best_score = grid_search.best_score_

            logging.info(f"Best parameters found: {best_params}")
            logging.info(f"Best cross-validation score: {best_score}")

            # Evaluate the best model on the validation set
            logging.info("Evaluating the best model on test data")
            best_clf = grid_search.best_estimator_
            y_pred = best_clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            logging.info(f"Accuracy of the Stacking Classifier: {accuracy:.2f}")
            print(f"Accuracy of the Stacking Classifier: {accuracy:.2f}")

            # Print classification report
            print(classification_report(y_test, y_pred))

            # Save the best model
            logging.info("Saving the best model")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_clf
            )

            return accuracy

        except Exception as e:
            raise CustomException(e, sys)
