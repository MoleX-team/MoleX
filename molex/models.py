"""
Model selection and configuration for MoleX.
"""

import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression, RidgeClassifier, LogisticRegressionCV, ElasticNetCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from pygam import LogisticGAM


class CompatibleClassifier(BaseEstimator, ClassifierMixin):
    """Wrapper to make GAM compatible with sklearn interface."""
    
    def __init__(self, base_model):
        self.base_model = base_model
    
    def fit(self, X, y):
        self.base_model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.base_model.predict(X)
    
    def predict_proba(self, X):
        if hasattr(self.base_model, 'predict_proba'):
            return self.base_model.predict_proba(X)
        else:
            # For models without predict_proba, create binary probabilities
            predictions = self.predict(X)
            n_classes = 2  # Assuming binary classification
            proba = np.zeros((len(predictions), n_classes))
            for i, pred in enumerate(predictions):
                proba[i, int(pred)] = 1.0
            return proba


class ModelSelector:
    """Model selection utility for MoleX."""
    
    @staticmethod
    def get_available_models():
        """Get list of available model names."""
        return list(ModelSelector._get_model_dict().keys())
    
    @staticmethod
    def _get_model_dict():
        """Get dictionary of available models."""
        return {
            "LogisticRegression": LogisticRegression(),
            "XGBoost": xgb.XGBClassifier(eval_metric='logloss'),
            "LDA": LinearDiscriminantAnalysis(),
            "QDA": QuadraticDiscriminantAnalysis(),
            "KNN": KNeighborsClassifier(),
            "NaiveBayes": GaussianNB(),
            "SVM": SVC(probability=True),
            "PolynomialRegression": make_pipeline(PolynomialFeatures(), LogisticRegression()),
            "GAM": CompatibleClassifier(LogisticGAM()),
            "StackedRegression": StackingClassifier(
                estimators=[('rf', RandomForestClassifier()), ('gb', GradientBoostingClassifier())],
                final_estimator=LogisticRegression()
            ),
            "RidgeRegression": RidgeClassifier(),
            "LassoRegression": LogisticRegressionCV(penalty='l1', solver='saga', cv=5),
            "ElasticNet": ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1]),
            "MultinomialLogisticRegression": OneVsRestClassifier(LogisticRegression()),
            "RandomForest": RandomForestClassifier(),
            "GBM": GradientBoostingClassifier(),
            "AdaBoost": AdaBoostClassifier()
        }
    
    @staticmethod
    def select_model(model_name):
        """
        Select and return a model instance.
        
        Args:
            model_name (str): Name of the model to select
            
        Returns:
            sklearn-compatible model instance
            
        Raises:
            ValueError: If model_name is not supported
        """
        models = ModelSelector._get_model_dict()
        
        if model_name in models:
            return models[model_name]
        else:
            available_models = list(models.keys())
            raise ValueError(f"Unsupported model: {model_name}. Available models: {available_models}")