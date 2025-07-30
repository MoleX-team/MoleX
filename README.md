# MoleX: Molecular Property Prediction Library

MoleX is a Python library for molecular property prediction using embedding-based machine learning approaches. It combines molecular embeddings with traditional machine learning models and residual learning techniques to achieve high-performance predictions on various molecular datasets.

## Features

- **Molecular Embedding**: Support for various molecular embedding models including SentenceTransformers
- **Multiple ML Models**: Integration with 17+ machine learning algorithms including XGBoost, Random Forest, SVM, etc.
- **Residual Learning**: Advanced residual modeling to improve prediction accuracy
- **Fragment Analysis**: N-gram based molecular fragment importance analysis
- **Easy-to-use API**: Simple and intuitive interface for molecular property prediction

## Installation


```bash
cd molex
pip install -e .
```

## Quick Start

```python
import molex
from molex import MolecularPredictor

# Initialize predictor
predictor = MolecularPredictor(
    embedding_model='your-embedding-model',
    main_model='LogisticRegression',
    residual_model='XGBoost'
)

# Load your data (SELFIES format)
train_selfies = ['[C][C][O]', '[C][C][C][O]', ...]  # Your training molecules
train_labels = [0, 1, 0, ...]  # Your training labels
test_selfies = ['[C][O]', '[C][C][C]', ...]  # Your test molecules

# Train and predict
predictor.fit(train_selfies, train_labels)
predictions = predictor.predict(test_selfies)
probabilities = predictor.predict_proba(test_selfies)

# Get performance metrics
accuracy = predictor.score(test_selfies, test_labels)
roc_auc = predictor.roc_auc_score(test_selfies, test_labels)
```

## Advanced Usage

### Fragment Analysis

```python
from molex import FragmentAnalyzer

analyzer = FragmentAnalyzer()
important_fragments = analyzer.get_important_fragments(train_selfies, train_labels)
print("Most important molecular fragments:", important_fragments)
```

### Custom Model Configuration

```python
predictor = MolecularPredictor(
    embedding_model='your-model',
    main_model='RandomForest',
    residual_model='XGBoost',
    reduced_dimension=60,
    residual_iterations=5
)
```

### Batch Processing

```python
from molex import BatchProcessor

processor = BatchProcessor()
results = processor.process_datasets(
    datasets=['BBBP', 'Tox21', 'SIDER'],
    embedding_model='your-model'
)
```

## Supported Models

### Main Models
- LogisticRegression
- XGBoost
- RandomForest
- SVM
- LinearDiscriminantAnalysis
- QuadraticDiscriminantAnalysis
- KNeighborsClassifier
- GaussianNB
- GradientBoostingClassifier
- AdaBoostClassifier
- And more...

### Residual Models
- XGBoost (default)
- RandomForest
- LogisticRegression
- All main models are supported
