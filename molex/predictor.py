"""
Main molecular predictor class for MoleX.
"""

import numpy as np
import time
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer

from .models import ModelSelector
from .utils import calculate_accuracy, calculate_roc_auc


class MolecularPredictor:
    """
    Main molecular property predictor using embedding-based machine learning.
    
    This class combines molecular embeddings with traditional ML models and 
    residual learning for improved prediction accuracy.
    """
    
    def __init__(self, embedding_model, main_model='LogisticRegression', 
                 residual_model='XGBoost', reduced_dimension=60, 
                 residual_iterations=5):
        """
        Initialize the molecular predictor.
        
        Args:
            embedding_model (str): Name or path of the embedding model
            main_model (str): Name of the main classification model
            residual_model (str): Name of the residual model
            reduced_dimension (int): Number of PCA components
            residual_iterations (int): Number of residual learning iterations
        """
        self.embedding_model_name = embedding_model
        self.main_model_name = main_model
        self.residual_model_name = residual_model
        self.reduced_dimension = reduced_dimension
        self.residual_iterations = residual_iterations
        
        # Initialize components
        self.embedding_model = None
        self.pca = None
        self.main_classifier = None
        self.residual_classifier = None
        
        # Training data storage for residual learning
        self.train_embeddings = None
        self.train_labels = None
        
    def _load_embedding_model(self):
        """Load the sentence transformer model."""
        if self.embedding_model is None:
            self.embedding_model = SentenceTransformer(
                self.embedding_model_name, 
                trust_remote_code=True
            )
    
    def _encode_molecules(self, molecules, show_progress=True):
        """
        Encode molecules using the embedding model.
        
        Args:
            molecules (list): List of SELFIES strings
            show_progress (bool): Whether to show progress bar
            
        Returns:
            numpy.ndarray: Encoded embeddings
        """
        self._load_embedding_model()
        return self.embedding_model.encode(molecules, show_progress_bar=show_progress)
    
    def _apply_pca(self, embeddings, fit=False):
        """
        Apply PCA dimensionality reduction.
        
        Args:
            embeddings (numpy.ndarray): Input embeddings
            fit (bool): Whether to fit PCA or just transform
            
        Returns:
            numpy.ndarray: Reduced embeddings
        """
        if fit:
            self.pca = PCA(n_components=self.reduced_dimension)
            return self.pca.fit_transform(embeddings)
        else:
            if self.pca is None:
                raise ValueError("PCA not fitted. Call fit() first.")
            return self.pca.transform(embeddings)
    
    def fit(self, train_molecules, train_labels):
        """
        Fit the molecular predictor.
        
        Args:
            train_molecules (list): List of training SELFIES strings
            train_labels (array-like): Training labels
        """
        # Encode molecules
        print("Encoding training molecules...")
        start_time = time.time()
        train_embeddings = self._encode_molecules(train_molecules)
        self.train_embedding_time = time.time() - start_time
        
        # Apply PCA
        self.train_embeddings = self._apply_pca(train_embeddings, fit=True)
        self.train_labels = np.array(train_labels)
        
        # Train main classifier
        self.main_classifier = ModelSelector.select_model(self.main_model_name)
        self.main_classifier.fit(self.train_embeddings, self.train_labels)
        
        print(f"Training completed in {self.train_embedding_time:.2f} seconds")
    
    def predict(self, test_molecules):
        """
        Predict molecular properties.
        
        Args:
            test_molecules (list): List of test SELFIES strings
            
        Returns:
            numpy.ndarray: Predicted labels
        """
        if self.main_classifier is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Encode and transform test molecules
        test_embeddings = self._encode_molecules(test_molecules, show_progress=False)
        test_embeddings = self._apply_pca(test_embeddings)
        
        # Get main predictions
        main_predictions = self.main_classifier.predict(test_embeddings)
        
        # Apply residual learning if enabled
        if self.residual_iterations > 0:
            return self._apply_residual_learning(
                test_embeddings, main_predictions, 
                self.main_classifier.predict_proba(test_embeddings)[:, 1]
            )
        
        return main_predictions
    
    def predict_proba(self, test_molecules):
        """
        Predict molecular property probabilities.
        
        Args:
            test_molecules (list): List of test SELFIES strings
            
        Returns:
            numpy.ndarray: Predicted probabilities
        """
        if self.main_classifier is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Encode and transform test molecules
        test_embeddings = self._encode_molecules(test_molecules, show_progress=False)
        test_embeddings = self._apply_pca(test_embeddings)
        
        # Get main predictions
        main_proba = self.main_classifier.predict_proba(test_embeddings)
        
        # Apply residual learning if enabled
        if self.residual_iterations > 0:
            main_predictions = main_proba.argmax(axis=1)
            final_proba = self._apply_residual_learning(
                test_embeddings, main_predictions, main_proba[:, 1], return_proba=True
            )
            return np.column_stack([1 - final_proba, final_proba])
        
        return main_proba
    
    def _apply_residual_learning(self, test_embeddings, main_predictions, main_proba, return_proba=False):
        """
        Apply residual learning to improve predictions.
        
        Args:
            test_embeddings (numpy.ndarray): Test embeddings
            main_predictions (numpy.ndarray): Main model predictions
            main_proba (numpy.ndarray): Main model probabilities
            return_proba (bool): Whether to return probabilities
            
        Returns:
            numpy.ndarray: Final predictions or probabilities
        """
        current_pred = main_predictions.copy()
        current_proba = main_proba.copy()
        
        for iteration in range(self.residual_iterations):
            try:
                # Generate residual training set based on training data
                residual_embeddings, residual_labels = self._generate_residual_training_set(None)
                
                if len(residual_embeddings) == 0:
                    print(f"  Warning: No residual training data available for iteration {iteration + 1}")
                    break
                
                # Train residual model
                residual_classifier = ModelSelector.select_model(self.residual_model_name)
                residual_classifier.fit(residual_embeddings, residual_labels)
                
                # Apply residual corrections to test data
                residual_pred = residual_classifier.predict(test_embeddings)
                
                # Combine predictions - only flip if residual model suggests correction
                correction_mask = residual_pred == 1
                current_pred[correction_mask] = 1 - current_pred[correction_mask]
                current_proba[correction_mask] = 1 - current_proba[correction_mask]
                
            except Exception as e:
                print(f"  Warning: Residual learning iteration {iteration + 1} failed: {str(e)}")
                break
        
        return current_proba if return_proba else (current_proba > 0.5).astype(int)
    
    def _generate_residual_training_set(self, test_predictions):
        """Generate balanced residual training set based on test predictions."""
        # We need to generate residual labels based on training data
        # First, get training predictions
        train_predictions = self.main_classifier.predict(self.train_embeddings)
        residual_labels = np.abs(self.train_labels - train_predictions)
        
        # Balance the dataset
        num_correct = np.sum(residual_labels == 0)
        num_incorrect = np.sum(residual_labels == 1)
        num_choose = min(num_correct, num_incorrect)
        
        if num_choose == 0:
            return np.array([]), np.array([])
        
        # Select balanced samples
        correct_indices = np.where(residual_labels == 0)[0][:num_choose]
        incorrect_indices = np.where(residual_labels == 1)[0][:num_choose]
        
        selected_indices = np.concatenate([correct_indices, incorrect_indices])
        
        return (self.train_embeddings[selected_indices], 
                residual_labels[selected_indices])
    
    
    def score(self, test_molecules, test_labels):
        """
        Calculate accuracy score.
        
        Args:
            test_molecules (list): List of test SELFIES strings
            test_labels (array-like): True labels
            
        Returns:
            float: Accuracy score
        """
        predictions = self.predict(test_molecules)
        return calculate_accuracy(test_labels, predictions)
    
    def roc_auc_score(self, test_molecules, test_labels):
        """
        Calculate ROC AUC score.
        
        Args:
            test_molecules (list): List of test SELFIES strings
            test_labels (array-like): True labels
            
        Returns:
            float: ROC AUC score
        """
        probabilities = self.predict_proba(test_molecules)[:, 1]
        return calculate_roc_auc(test_labels, probabilities)