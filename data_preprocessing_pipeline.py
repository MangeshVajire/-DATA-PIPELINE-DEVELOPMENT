
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

class OptimizationDataPipeline:
    def __init__(self):
        self.numerical_features = None
        self.categorical_features = None
        self.preprocessor = None
        self.scaler_type = 'standard'
        
    def create_pipeline(self, numerical_features, categorical_features, scaler_type='standard'):
        """
        Create preprocessing pipeline for optimization-related data
        """
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        self.scaler_type = scaler_type
        
        # Select scaler based on the type
        scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler()
        
        # Numerical pipeline with bounds-friendly scaling
        numerical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', scaler)
        ])
        
        # Categorical pipeline - replaced LabelEncoder with OneHotEncoder
        # Update the categorical pipeline with correct parameter
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
            ('encoder', OneHotEncoder(drop='first', sparse_output=False))
        ])
        
        # Combine pipelines
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_pipeline, numerical_features),
                ('cat', categorical_pipeline, categorical_features)
            ])
        
        return self.preprocessor
    
    def fit_transform_data(self, X):
        """
        Fit and transform the data
        """
        if self.preprocessor is None:
            raise ValueError("Pipeline not created. Call create_pipeline first.")
        
        transformed_data = self.preprocessor.fit_transform(X)
        
        # Get feature names for categorical columns after one-hot encoding
        cat_feature_names = []
        if self.categorical_features:
            encoder = self.preprocessor.named_transformers_['cat'].named_steps['encoder']
            cat_feature_names = encoder.get_feature_names_out(self.categorical_features)
        
        # Combine column names
        all_feature_names = list(self.numerical_features) + list(cat_feature_names)
        
        return pd.DataFrame(
            transformed_data,
            columns=all_feature_names
        )
    
    def transform_data(self, X):
        """
        Transform new data using fitted pipeline
        """
        if self.preprocessor is None:
            raise ValueError("Pipeline not created. Call create_pipeline first.")
        
        transformed_data = self.preprocessor.transform(X)
        
        # Get feature names for categorical columns after one-hot encoding
        cat_feature_names = []
        if self.categorical_features:
            encoder = self.preprocessor.named_transformers_['cat'].named_steps['encoder']
            cat_feature_names = encoder.get_feature_names_out(self.categorical_features)
        
        # Combine column names
        all_feature_names = list(self.numerical_features) + list(cat_feature_names)
        
        return pd.DataFrame(
            transformed_data,
            columns=all_feature_names
        )

# Example usage
if __name__ == "__main__":
    # Sample optimization-related data
    data = pd.DataFrame({
        'resource_capacity': [100, 150, np.nan, 200, 120],
        'production_rate': [10, 15, 20, np.nan, 12],
        'resource_type': ['labor', 'machine', 'labor', 'machine', np.nan],
        'priority': ['high', 'medium', np.nan, 'low', 'high']
    })
    
    # Initialize pipeline
    pipeline = OptimizationDataPipeline()
    
    # Define features
    numerical_cols = ['resource_capacity', 'production_rate']
    categorical_cols = ['resource_type', 'priority']
    
    # Create and fit pipeline
    preprocessor = pipeline.create_pipeline(
        numerical_cols, 
        categorical_cols, 
        scaler_type='minmax'  # Using MinMaxScaler for bounded optimization
    )
    
    transformed_data = pipeline.fit_transform_data(data)
    
    print("Original data shape:", data.shape)
    print("Transformed data shape:", transformed_data.shape)
    print("\nTransformed data preview:")
    print(transformed_data.head())
