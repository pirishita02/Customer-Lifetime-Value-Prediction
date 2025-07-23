import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    """Handle data preprocessing tasks including missing values, encoding, and scaling."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.imputers = {}
    
    def handle_missing_values(self, df):
        """Handle missing values in the dataset."""
        df_processed = df.copy()
        
        # First, try to convert object columns that might be numeric
        for col in df_processed.columns:
            if df_processed[col].dtype == 'object':
                # Try to convert to numeric
                numeric_col = pd.to_numeric(df_processed[col], errors='coerce')
                # If more than 50% of values are numeric, convert the column
                if numeric_col.notna().sum() / len(df_processed) > 0.5:
                    df_processed[col] = numeric_col
        
        # Separate numerical and categorical columns after conversion
        numerical_cols = df_processed.select_dtypes(include=[np.number]).columns
        categorical_cols = df_processed.select_dtypes(include=['object']).columns
        
        # Handle numerical missing values
        if len(numerical_cols) > 0:
            # Use median for numerical columns
            num_imputer = SimpleImputer(strategy='median')
            df_processed[numerical_cols] = num_imputer.fit_transform(df_processed[numerical_cols])
            self.imputers['numerical'] = num_imputer
        
        # Handle categorical missing values
        if len(categorical_cols) > 0:
            # Convert all values to strings first to ensure consistency
            for col in categorical_cols:
                df_processed[col] = df_processed[col].astype(str)
            
            # Use most frequent for categorical columns
            cat_imputer = SimpleImputer(strategy='most_frequent')
            df_processed[categorical_cols] = cat_imputer.fit_transform(df_processed[categorical_cols])
            self.imputers['categorical'] = cat_imputer
        
        return df_processed
    
    def encode_categorical_variables(self, df):
        """Encode categorical variables using label encoding."""
        df_processed = df.copy()
        categorical_cols = df_processed.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col not in ['CustomerID', 'InvoiceDate']:  # Skip ID and date columns
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                self.label_encoders[col] = le
        
        return df_processed
    
    def scale_features(self, df):
        """Scale numerical features using StandardScaler."""
        df_processed = df.copy()
        
        # Identify numerical columns (exclude IDs and encoded categoricals if needed)
        numerical_cols = df_processed.select_dtypes(include=[np.number]).columns
        
        # Exclude certain columns from scaling if they exist
        exclude_cols = ['CustomerID', 'R_Score', 'F_Score', 'M_Score', 'RFM_Score']
        numerical_cols = [col for col in numerical_cols if col not in exclude_cols]
        
        if len(numerical_cols) > 0:
            # Clean and validate numerical data
            for col in numerical_cols:
                # Convert to numeric
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                
                # Replace infinite values with NaN
                df_processed[col] = df_processed[col].replace([np.inf, -np.inf], np.nan)
                
                # Cap extremely large values before scaling
                col_median = df_processed[col].median()
                col_std = df_processed[col].std()
                
                if pd.notna(col_std) and col_std > 0:
                    # Cap values beyond 5 standard deviations
                    upper_bound = col_median + 5 * col_std
                    lower_bound = col_median - 5 * col_std
                    df_processed.loc[df_processed[col] > upper_bound, col] = upper_bound
                    df_processed.loc[df_processed[col] < lower_bound, col] = lower_bound
                
                # Fill remaining NaN values with median
                df_processed[col] = df_processed[col].fillna(col_median if pd.notna(col_median) else 0)
                
                # Final safety check for infinite values
                if np.isinf(df_processed[col]).any():
                    df_processed[col] = df_processed[col].replace([np.inf, -np.inf], col_median if pd.notna(col_median) else 0)
            
            # Verify no infinite values before scaling
            for col in numerical_cols:
                if np.isinf(df_processed[col]).any() or (df_processed[col].abs() > 1e10).any():
                    print(f"Warning: Large/infinite values detected in {col} before scaling")
                    df_processed[col] = np.clip(df_processed[col], -1e6, 1e6)
            
            # Apply scaling
            try:
                df_processed[numerical_cols] = self.scaler.fit_transform(df_processed[numerical_cols])
            except ValueError as e:
                print(f"Scaling error: {e}")
                # Fallback: manual min-max scaling
                for col in numerical_cols:
                    col_min = df_processed[col].min()
                    col_max = df_processed[col].max()
                    if col_max != col_min:
                        df_processed[col] = (df_processed[col] - col_min) / (col_max - col_min)
                    else:
                        df_processed[col] = 0
        
        return df_processed
    
    def detect_outliers(self, df, columns=None):
        """Detect outliers using IQR method."""
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns
        
        outliers_dict = {}
        
        for col in columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index
            outliers_dict[col] = outliers.tolist()
        
        return outliers_dict
    
    def remove_outliers(self, df, columns=None, method='iqr'):
        """Remove outliers from the dataset."""
        df_processed = df.copy()
        
        if method == 'iqr':
            outliers_dict = self.detect_outliers(df_processed, columns)
            
            # Get all outlier indices
            all_outliers = set()
            for outliers in outliers_dict.values():
                all_outliers.update(outliers)
            
            # Remove outliers
            df_processed = df_processed.drop(index=list(all_outliers))
            df_processed = df_processed.reset_index(drop=True)
        
        return df_processed
    
    def create_interaction_features(self, df, feature_pairs):
        """Create interaction features between specified feature pairs."""
        df_processed = df.copy()
        
        for feature1, feature2 in feature_pairs:
            if feature1 in df_processed.columns and feature2 in df_processed.columns:
                interaction_name = f"{feature1}_x_{feature2}"
                df_processed[interaction_name] = df_processed[feature1] * df_processed[feature2]
        
        return df_processed
    
    def bin_numerical_features(self, df, feature_bins):
        """Bin numerical features into categories."""
        df_processed = df.copy()
        
        for feature, bins in feature_bins.items():
            if feature in df_processed.columns:
                df_processed[f"{feature}_binned"] = pd.cut(df_processed[feature], 
                                                         bins=bins, 
                                                         labels=False)
        
        return df_processed
    
    def validate_data(self, df):
        """Validate the dataset for common issues."""
        issues = []
        
        # Check for empty dataset
        if df.empty:
            issues.append("Dataset is empty")
        
        # Check for missing values
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            issues.append(f"Dataset has {missing_count} missing values")
        
        # Check for duplicate rows
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            issues.append(f"Dataset has {duplicate_count} duplicate rows")
        
        # Check for constant columns
        constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
        if constant_cols:
            issues.append(f"Constant columns found: {constant_cols}")
        
        # Check data types
        object_cols = df.select_dtypes(include=['object']).columns
        for col in object_cols:
            if col not in ['CustomerID', 'InvoiceDate']:
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio > 0.8:
                    issues.append(f"High cardinality in column '{col}': {df[col].nunique()} unique values")
        
        return issues
