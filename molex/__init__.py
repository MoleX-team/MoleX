"""
MoleX: Molecular Property Prediction Library

A Python library for molecular property prediction using embedding-based machine learning.
"""

__version__ = "0.1.0"
__author__ = "MoleX Team"
__email__ = "contact@molex.ai"

from .predictor import MolecularPredictor
from .fragment_analyzer import FragmentAnalyzer
from .batch_processor import BatchProcessor
from .models import ModelSelector
from .utils import calculate_accuracy, calculate_roc_auc

__all__ = [
    "MolecularPredictor",
    "FragmentAnalyzer", 
    "BatchProcessor",
    "ModelSelector",
    "calculate_accuracy",
    "calculate_roc_auc"
]