"""
Fragment analysis for molecular interpretability.
"""

import re
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression


class FragmentAnalyzer:
    """
    Analyzer for molecular fragment importance using n-gram analysis.
    
    This class extracts molecular fragments from SELFIES strings and 
    determines their importance for property prediction.
    """
    
    def __init__(self, ngram_range=(1, 3), min_df=1, max_fragments=30):
        """
        Initialize the fragment analyzer.
        
        Args:
            ngram_range (tuple): Range of n-grams to extract
            min_df (int): Minimum document frequency for n-grams
            max_fragments (int): Maximum number of fragments to consider
        """
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_fragments = max_fragments
        
    def _molecular_tokenizer(self, text):
        """
        Custom tokenizer for molecular fragments in SELFIES format.
        
        Args:
            text (str): SELFIES string
            
        Returns:
            list: List of molecular tokens
        """
        return re.findall(r'\[.*?\]', text)
    
    def get_important_fragments(self, molecules, labels, top_k=5):
        """
        Get the most important molecular fragments for prediction.
        
        Args:
            molecules (list): List of SELFIES strings
            labels (array-like): Corresponding labels
            top_k (int): Number of top fragments to return
            
        Returns:
            list: List of important fragments with their weights
        """
        # Create vectorizer for n-gram extraction
        vectorizer = CountVectorizer(
            tokenizer=self._molecular_tokenizer,
            ngram_range=self.ngram_range,
            min_df=self.min_df
        )
        
        # Extract features
        X = vectorizer.fit_transform(molecules)
        
        # Train logistic regression for feature importance
        model = LogisticRegression(max_iter=1000)
        model.fit(X, labels)
        
        # Get feature names and coefficients
        feature_names = vectorizer.get_feature_names_out()
        coefficients = model.coef_[0]
        
        # Process n-grams to extract fragment weights
        fragment_weights = self._extract_fragment_weights(feature_names, coefficients)
        
        # Return top fragments by absolute weight
        sorted_fragments = sorted(
            fragment_weights.items(), 
            key=lambda x: abs(x[1]), 
            reverse=True
        )
        
        return [f'{frag}: {weight:.3f}' for frag, weight in sorted_fragments[:top_k]]
    
    def _extract_fragment_weights(self, feature_names, coefficients):
        """
        Extract fragment weights from n-gram coefficients.
        
        Args:
            feature_names (array): N-gram feature names
            coefficients (array): Model coefficients
            
        Returns:
            dict: Fragment weights
        """
        fragment_weights = defaultdict(float)
        
        # Create mapping from n-grams to coefficients
        ngram_weights = dict(zip(feature_names, coefficients))
        
        # Process each n-gram to extract fragment information
        for ngram, weight in ngram_weights.items():
            # Extract fragments from n-gram
            fragments = re.findall(r'\[([^\]]+)\]', ngram)
            
            for fragment in fragments:
                # Check if it's a numbered fragment
                if (match := re.search(r'frag(\d+)', fragment)):
                    frag_name = f'frag{match.group(1)}'
                    fragment_weights[frag_name] += weight
                else:
                    # Use the fragment as is
                    fragment_weights[f'[{fragment}]'] += weight
        
        # Add zero weights for missing numbered fragments
        for i in range(self.max_fragments):
            frag_name = f'frag{i}'
            if frag_name not in fragment_weights:
                fragment_weights[frag_name] = 0.0
        
        return dict(fragment_weights)
    
    def analyze_molecule(self, molecule, model_weights=None):
        """
        Analyze a single molecule for fragment contributions.
        
        Args:
            molecule (str): SELFIES string
            model_weights (dict): Pre-computed fragment weights
            
        Returns:
            dict: Fragment analysis results
        """
        fragments = self._molecular_tokenizer(molecule)
        
        if model_weights is None:
            return {'fragments': fragments, 'weights': None}
        
        # Calculate weighted contribution
        total_weight = 0
        fragment_contributions = {}
        
        for fragment in fragments:
            weight = model_weights.get(fragment, 0.0)
            fragment_contributions[fragment] = weight
            total_weight += weight
        
        return {
            'fragments': fragments,
            'fragment_contributions': fragment_contributions,
            'total_weight': total_weight
        }