"""
Batch processing utilities for multiple datasets.
"""

import numpy as np
import datasets
from .predictor import MolecularPredictor
from .utils import calculate_statistics, handle_nan_labels


class BatchProcessor:
    """
    Batch processor for handling multiple molecular datasets.
    
    This class provides utilities for processing multiple datasets
    and comparing model performance across different molecular properties.
    """
    
    def __init__(self, embedding_model, reduced_dimension=60, 
                 residual_iterations=1, num_runs=3):
        """
        Initialize the batch processor.
        
        Args:
            embedding_model (str): Name or path of embedding model
            reduced_dimension (int): PCA dimensions
            residual_iterations (int): Residual learning iterations
            num_runs (int): Number of runs for averaging results
        """
        self.embedding_model = embedding_model
        self.reduced_dimension = reduced_dimension
        self.residual_iterations = residual_iterations
        self.num_runs = num_runs
    
    def load_dataset(self, dataset_name, selfies_column='group_selfies', 
                     label_column=None, test_size=0.2):
        """
        Load a molecular dataset.
        
        Args:
            dataset_name (str): Name of the dataset file (without .csv)
            selfies_column (str): Name of the SELFIES column
            label_column (str): Name of the label column
            test_size (float): Fraction of data for testing
            
        Returns:
            tuple: (train_molecules, test_molecules, train_labels, test_labels)
        """
        try:
            # Load dataset
            dataset = datasets.load_dataset("csv", data_files=f"{dataset_name}.csv")["train"]
            split_dataset = dataset.train_test_split(test_size=test_size)
            
            train_dataset = split_dataset['train']
            test_dataset = split_dataset['test']
            
            # Extract molecules
            train_molecules = train_dataset[selfies_column]
            test_molecules = test_dataset[selfies_column]
            
            # Extract and process labels
            train_labels = np.array(train_dataset[label_column])
            test_labels = np.array(test_dataset[label_column])
            
            # Handle NaN values
            train_labels = handle_nan_labels(train_labels)
            test_labels = handle_nan_labels(test_labels)
            
            return train_molecules, test_molecules, train_labels, test_labels
            
        except Exception as e:
            raise ValueError(f"Error loading dataset {dataset_name}: {str(e)}")
    
    def process_single_dataset(self, dataset_name, label_columns, 
                              main_model='LogisticRegression', 
                              residual_model='XGBoost'):
        """
        Process a single dataset with multiple label columns.
        
        Args:
            dataset_name (str): Name of the dataset
            label_columns (list): List of label column names
            main_model (str): Main model name
            residual_model (str): Residual model name
            
        Returns:
            dict: Results for each label column
        """
        results = {}
        
        print(f"Processing {dataset_name} dataset...")
        
        for label_column in label_columns:
            print(f"  Processing label: {label_column}")
            
            try:
                # Run multiple times for statistical significance
                accuracies = []
                roc_aucs = []
                final_accuracies = []
                final_roc_aucs = []
                
                for run in range(self.num_runs):
                    # Load data
                    train_mol, test_mol, train_lab, test_lab = self.load_dataset(
                        dataset_name, label_column=label_column
                    )
                    
                    # Create and train predictor
                    predictor = MolecularPredictor(
                        embedding_model=self.embedding_model,
                        main_model=main_model,
                        residual_model=residual_model,
                        reduced_dimension=self.reduced_dimension,
                        residual_iterations=self.residual_iterations
                    )
                    
                    predictor.fit(train_mol, train_lab)
                    
                    # Get main model performance
                    main_accuracy = predictor.score(test_mol, test_lab)
                    main_roc_auc = predictor.roc_auc_score(test_mol, test_lab)
                    
                    accuracies.append(main_accuracy)
                    roc_aucs.append(main_roc_auc)
                    
                    # Get final performance (with residual learning)
                    if self.residual_iterations > 0:
                        final_predictions = predictor.predict(test_mol)
                        final_probabilities = predictor.predict_proba(test_mol)[:, 1]
                        
                        final_accuracy = np.mean(test_lab == final_predictions)
                        final_roc_auc = predictor.roc_auc_score(test_mol, test_lab)
                        
                        final_accuracies.append(final_accuracy)
                        final_roc_aucs.append(final_roc_auc)
                    else:
                        final_accuracies.append(main_accuracy)
                        final_roc_aucs.append(main_roc_auc)
                
                # Calculate statistics
                acc_mean, acc_std = calculate_statistics(accuracies)
                roc_mean, roc_std = calculate_statistics(roc_aucs)
                final_acc_mean, final_acc_std = calculate_statistics(final_accuracies)
                final_roc_mean, final_roc_std = calculate_statistics(final_roc_aucs)
                
                results[label_column] = {
                    'accuracy': (acc_mean * 100, acc_std * 100),
                    'roc_auc': (roc_mean * 100, roc_std * 100),
                    'final_accuracy': (final_acc_mean * 100, final_acc_std * 100),
                    'final_roc_auc': (final_roc_mean * 100, final_roc_std * 100)
                }
                
                print(f"    Accuracy: {acc_mean*100:.2f} ± {acc_std*100:.2f}")
                print(f"    ROC AUC: {roc_mean*100:.2f} ± {roc_std*100:.2f}")
                print(f"    Final Accuracy: {final_acc_mean*100:.2f} ± {final_acc_std*100:.2f}")
                print(f"    Final ROC AUC: {final_roc_mean*100:.2f} ± {final_roc_std*100:.2f}")
                
            except Exception as e:
                print(f"    Error processing {label_column}: {str(e)}")
                results[label_column] = {'error': str(e)}
        
        return results
    
    def process_multiple_datasets(self, datasets_config, 
                                 main_model='LogisticRegression',
                                 residual_model='XGBoost'):
        """
        Process multiple datasets.
        
        Args:
            datasets_config (dict): Dictionary mapping dataset names to label columns
            main_model (str): Main model name
            residual_model (str): Residual model name
            
        Returns:
            dict: Results for all datasets
        """
        all_results = {}
        
        for dataset_name, label_columns in datasets_config.items():
            all_results[dataset_name] = self.process_single_dataset(
                dataset_name, label_columns, main_model, residual_model
            )
        
        return all_results
    
    def print_summary(self, results):
        """
        Print a formatted summary of results.
        
        Args:
            results (dict): Results from process_multiple_datasets
        """
        print("\n" + "="*80)
        print("MOLECULAR PROPERTY PREDICTION RESULTS SUMMARY")
        print("="*80)
        
        for dataset_name, dataset_results in results.items():
            print(f"\n{dataset_name} Results:")
            print("-" * 60)
            print(f"{'Label':<40} {'Accuracy':<15} {'ROC AUC':<15} {'Final Acc':<15} {'Final AUC':<15}")
            print("-" * 60)
            
            for label, metrics in dataset_results.items():
                if 'error' in metrics:
                    print(f"{label:<40} ERROR: {metrics['error']}")
                else:
                    acc = f"{metrics['accuracy'][0]:.1f}±{metrics['accuracy'][1]:.1f}"
                    roc = f"{metrics['roc_auc'][0]:.1f}±{metrics['roc_auc'][1]:.1f}"
                    final_acc = f"{metrics['final_accuracy'][0]:.1f}±{metrics['final_accuracy'][1]:.1f}"
                    final_roc = f"{metrics['final_roc_auc'][0]:.1f}±{metrics['final_roc_auc'][1]:.1f}"
                    
                    print(f"{label:<40} {acc:<15} {roc:<15} {final_acc:<15} {final_roc:<15}")
        
        print("\n" + "="*80)