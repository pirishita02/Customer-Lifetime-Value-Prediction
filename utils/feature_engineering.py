import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """Create advanced features for Customer Lifetime Value prediction."""
    
    def __init__(self):
        self.reference_date = None
    
    def create_rfm_features(self, df):
        """Create Recency, Frequency, and Monetary features."""
        if 'CustomerID' not in df.columns:
            raise ValueError("CustomerID column is required for RFM analysis")
        
        # Create a copy to avoid modifying original data
        df_copy = df.copy()
        
        # Ensure InvoiceDate is datetime
        if 'InvoiceDate' in df_copy.columns:
            df_copy['InvoiceDate'] = pd.to_datetime(df_copy['InvoiceDate'], errors='coerce')
            # Remove rows with invalid dates
            df_copy = df_copy.dropna(subset=['InvoiceDate'])
            self.reference_date = df_copy['InvoiceDate'].max()
        else:
            # If no date column, create a dummy reference date
            self.reference_date = datetime.now()
            df_copy['InvoiceDate'] = self.reference_date - timedelta(days=np.random.randint(1, 365, len(df_copy)))
        
        # Ensure TotalAmount exists and is numeric
        if 'TotalAmount' not in df_copy.columns:
            if 'Quantity' in df_copy.columns and 'UnitPrice' in df_copy.columns:
                # Convert to numeric first
                df_copy['Quantity'] = pd.to_numeric(df_copy['Quantity'], errors='coerce')
                df_copy['UnitPrice'] = pd.to_numeric(df_copy['UnitPrice'], errors='coerce')
                df_copy['TotalAmount'] = df_copy['Quantity'] * df_copy['UnitPrice']
            else:
                # Create a dummy TotalAmount based on available data
                numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    df_copy['TotalAmount'] = pd.to_numeric(df_copy[numeric_cols[0]], errors='coerce')
                else:
                    df_copy['TotalAmount'] = np.random.uniform(10, 1000, len(df_copy))
        else:
            # Ensure TotalAmount is numeric
            df_copy['TotalAmount'] = pd.to_numeric(df_copy['TotalAmount'], errors='coerce')
        
        # Remove rows with missing or negative values
        df_copy = df_copy.dropna(subset=['TotalAmount'])
        df_copy = df_copy[df_copy['TotalAmount'] > 0]
        
        # Calculate RFM metrics
        rfm = df_copy.groupby('CustomerID').agg({
            'InvoiceDate': lambda x: (self.reference_date - x.max()).days,  # Recency
            'TotalAmount': ['count', 'sum']  # Frequency and Monetary
        }).reset_index()
        
        # Handle multi-level column names properly
        if rfm.columns.nlevels > 1:
            # Flatten multi-level columns
            new_columns = []
            for col in rfm.columns:
                if col[0] == 'CustomerID':
                    new_columns.append('CustomerID')
                elif col[0] == 'InvoiceDate':
                    new_columns.append('Recency')
                elif col[0] == 'TotalAmount' and col[1] == 'count':
                    new_columns.append('Frequency')
                elif col[0] == 'TotalAmount' and col[1] == 'sum':
                    new_columns.append('Monetary')
                else:
                    new_columns.append('_'.join([str(c) for c in col if c != '']))
            rfm.columns = new_columns
        else:
            # Single level columns (fallback)
            rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
        
        # Ensure all metrics are positive and numeric
        rfm['Recency'] = pd.to_numeric(rfm['Recency'], errors='coerce').fillna(0)
        rfm['Frequency'] = pd.to_numeric(rfm['Frequency'], errors='coerce').fillna(1)
        rfm['Monetary'] = pd.to_numeric(rfm['Monetary'], errors='coerce').fillna(0)
        
        # Ensure minimum values
        rfm['Recency'] = np.maximum(rfm['Recency'], 0)
        rfm['Frequency'] = np.maximum(rfm['Frequency'], 1)
        rfm['Monetary'] = np.maximum(rfm['Monetary'], 0)
        
        # Add additional RFM-derived features
        rfm['AvgOrderValue'] = rfm['Monetary'] / rfm['Frequency']
        
        # Calculate customer tenure (time since first purchase)
        first_purchase = df_copy.groupby('CustomerID')['InvoiceDate'].min().reset_index()
        first_purchase.columns = ['CustomerID', 'FirstPurchase']
        rfm = rfm.merge(first_purchase, on='CustomerID')
        rfm['Tenure'] = (self.reference_date - rfm['FirstPurchase']).dt.days
        rfm['Tenure'] = pd.to_numeric(rfm['Tenure'], errors='coerce').fillna(30)
        rfm['Tenure'] = np.maximum(rfm['Tenure'], 1)  # Ensure positive tenure
        
        # Create additional features
        rfm = self.add_advanced_features(rfm, df_copy)
        
        return rfm
    
    def add_advanced_features(self, rfm_df, transaction_df):
        """Add advanced features to the RFM dataset."""
        # Calculate purchase intervals
        purchase_intervals = transaction_df.groupby('CustomerID').apply(
            lambda x: x['InvoiceDate'].sort_values().diff().dt.days.mean()
        ).reset_index()
        purchase_intervals.columns = ['CustomerID', 'AvgPurchaseInterval']
        rfm_df = rfm_df.merge(purchase_intervals, on='CustomerID', how='left')
        
        # Fill NaN values in AvgPurchaseInterval with median
        rfm_df['AvgPurchaseInterval'] = rfm_df['AvgPurchaseInterval'].fillna(
            rfm_df['AvgPurchaseInterval'].median()
        )
        
        # Calculate seasonal patterns (if enough data)
        if len(transaction_df) > 100:
            seasonal_patterns = transaction_df.copy()
            seasonal_patterns['Month'] = seasonal_patterns['InvoiceDate'].dt.month
            seasonal_patterns['Quarter'] = seasonal_patterns['InvoiceDate'].dt.quarter
            seasonal_patterns['DayOfWeek'] = seasonal_patterns['InvoiceDate'].dt.dayofweek
            
            # Monthly spending pattern
            monthly_spend = seasonal_patterns.groupby(['CustomerID', 'Month'])['TotalAmount'].sum().reset_index()
            monthly_variability = monthly_spend.groupby('CustomerID')['TotalAmount'].std().reset_index()
            monthly_variability.columns = ['CustomerID', 'MonthlySpendVariability']
            rfm_df = rfm_df.merge(monthly_variability, on='CustomerID', how='left')
            rfm_df['MonthlySpendVariability'] = rfm_df['MonthlySpendVariability'].fillna(0)
        else:
            rfm_df['MonthlySpendVariability'] = 0
        
        # Calculate RFM scores
        rfm_df = self.calculate_rfm_scores(rfm_df)
        
        return rfm_df
    
    def calculate_rfm_scores(self, rfm_df):
        """Calculate RFM scores (1-5 scale for each dimension)."""
        # Ensure numeric data types
        rfm_df['Recency'] = pd.to_numeric(rfm_df['Recency'], errors='coerce')
        rfm_df['Frequency'] = pd.to_numeric(rfm_df['Frequency'], errors='coerce')
        rfm_df['Monetary'] = pd.to_numeric(rfm_df['Monetary'], errors='coerce')
        
        # Fill any NaN values with median
        rfm_df['Recency'] = rfm_df['Recency'].fillna(rfm_df['Recency'].median())
        rfm_df['Frequency'] = rfm_df['Frequency'].fillna(1)
        rfm_df['Monetary'] = rfm_df['Monetary'].fillna(rfm_df['Monetary'].median())
        
        # Use percentile-based scoring instead of qcut to avoid ties issue
        try:
            # Recency Score (lower recency = higher score)
            rfm_df['R_Score'] = pd.qcut(rfm_df['Recency'].rank(method='first'), 5, labels=False, duplicates='drop') + 1
            rfm_df['R_Score'] = 6 - rfm_df['R_Score']  # Reverse so lower recency = higher score
            
            # Frequency Score (higher frequency = higher score)
            rfm_df['F_Score'] = pd.qcut(rfm_df['Frequency'].rank(method='first'), 5, labels=False, duplicates='drop') + 1
            
            # Monetary Score (higher monetary = higher score)
            rfm_df['M_Score'] = pd.qcut(rfm_df['Monetary'].rank(method='first'), 5, labels=False, duplicates='drop') + 1
            
        except ValueError:
            # Fallback to percentile-based scoring if qcut fails
            rfm_df['R_Score'] = pd.cut(rfm_df['Recency'], bins=5, labels=False, duplicates='drop') + 1
            rfm_df['R_Score'] = 6 - rfm_df['R_Score']  # Reverse so lower recency = higher score
            
            rfm_df['F_Score'] = pd.cut(rfm_df['Frequency'], bins=5, labels=False, duplicates='drop') + 1
            rfm_df['M_Score'] = pd.cut(rfm_df['Monetary'], bins=5, labels=False, duplicates='drop') + 1
        
        # Ensure scores are integers and handle any NaN values
        rfm_df['R_Score'] = rfm_df['R_Score'].fillna(3).astype(int)
        rfm_df['F_Score'] = rfm_df['F_Score'].fillna(3).astype(int)
        rfm_df['M_Score'] = rfm_df['M_Score'].fillna(3).astype(int)
        
        # Calculate combined RFM Score
        rfm_df['RFM_Score'] = rfm_df['R_Score'] + rfm_df['F_Score'] + rfm_df['M_Score']
        
        return rfm_df
    
    def calculate_clv(self, rfm_df):
        """Calculate Customer Lifetime Value using multiple approaches."""
        # Traditional CLV calculation
        # CLV = (Average Order Value × Purchase Frequency × Customer Lifespan)
        
        # Clean and validate input data first
        rfm_df = rfm_df.copy()
        
        # Replace any infinite or extremely large values
        numeric_cols = rfm_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            # Replace inf values with NaN, then fill with reasonable defaults
            rfm_df[col] = rfm_df[col].replace([np.inf, -np.inf], np.nan)
            
            # Cap extremely large values (> 1e6)
            rfm_df.loc[rfm_df[col] > 1e6, col] = 1e6
            
            # Fill NaN values with column median or default
            if col in ['Frequency']:
                rfm_df[col] = rfm_df[col].fillna(1)
            elif col in ['Tenure', 'AvgPurchaseInterval']:
                rfm_df[col] = rfm_df[col].fillna(30)
            else:
                rfm_df[col] = rfm_df[col].fillna(rfm_df[col].median())
        
        # Ensure minimum values for calculations
        rfm_df['Frequency'] = np.maximum(rfm_df['Frequency'], 1)
        rfm_df['Tenure'] = np.maximum(rfm_df['Tenure'], 1)
        rfm_df['AvgPurchaseInterval'] = np.maximum(rfm_df['AvgPurchaseInterval'], 1)
        rfm_df['AvgOrderValue'] = np.maximum(rfm_df['AvgOrderValue'], 0.01)
        
        # Estimate customer lifespan based on purchase frequency and recency
        # Higher frequency and lower recency suggest longer lifespan
        rfm_df['EstimatedLifespan'] = np.where(
            rfm_df['Frequency'] > 1,
            np.minimum((rfm_df['Tenure'] / rfm_df['Frequency']) * rfm_df['Frequency'] * 2, 365 * 5),  # Cap at 5 years
            365  # Default to 1 year for new customers
        )
        
        # Ensure reasonable lifespan range (30 days to 5 years)
        rfm_df['EstimatedLifespan'] = np.clip(rfm_df['EstimatedLifespan'], 30, 365 * 5)
        
        # Calculate CLV using traditional formula with safety checks
        purchase_frequency_annual = np.maximum(
            rfm_df['EstimatedLifespan'] / rfm_df['AvgPurchaseInterval'], 
            0.1  # Minimum frequency
        )
        
        rfm_df['CLV_Traditional'] = (
            rfm_df['AvgOrderValue'] * 
            purchase_frequency_annual *
            np.where(rfm_df['Frequency'] > 1, 1, 0.5)  # Discount for single-purchase customers
        )
        
        # Cap traditional CLV at reasonable maximum
        rfm_df['CLV_Traditional'] = np.minimum(rfm_df['CLV_Traditional'], rfm_df['Monetary'] * 50)
        
        # Create a more sophisticated CLV metric
        # Consider customer engagement and loyalty
        engagement_multiplier = (
            (rfm_df['R_Score'] * 0.3) +  # Recent activity
            (rfm_df['F_Score'] * 0.4) +  # Purchase frequency
            (rfm_df['M_Score'] * 0.3)    # Monetary value
        ) / 5  # Normalize to 0-1
        
        # Ensure engagement multiplier is reasonable
        engagement_multiplier = np.clip(engagement_multiplier, 0.1, 2.0)
        
        rfm_df['CLV'] = rfm_df['CLV_Traditional'] * engagement_multiplier
        
        # Ensure CLV is positive and reasonable
        rfm_df['CLV'] = np.maximum(rfm_df['CLV'], rfm_df['Monetary'])  # CLV should be at least current monetary value
        rfm_df['CLV'] = np.minimum(rfm_df['CLV'], rfm_df['Monetary'] * 100)  # Cap at 100x current value
        
        # Final safety check for infinite or extremely large values
        rfm_df['CLV'] = rfm_df['CLV'].replace([np.inf, -np.inf], rfm_df['Monetary'])
        rfm_df['CLV'] = np.where(rfm_df['CLV'] > 1e6, rfm_df['Monetary'] * 10, rfm_df['CLV'])
        
        return rfm_df
    
    def create_customer_segments(self, rfm_df):
        """Create customer segments based on RFM scores."""
        def segment_customers(row):
            """Segment customers based on RFM scores."""
            if row['RFM_Score'] >= 12:
                return 'Champions'
            elif row['RFM_Score'] >= 9:
                if row['R_Score'] >= 4:
                    return 'Loyal Customers'
                else:
                    return 'Potential Loyalists'
            elif row['RFM_Score'] >= 6:
                if row['R_Score'] >= 3:
                    return 'New Customers'
                else:
                    return 'At Risk'
            else:
                if row['R_Score'] <= 2:
                    return 'Lost'
                else:
                    return 'Cannot Lose Them'
        
        rfm_df['CustomerSegment'] = rfm_df.apply(segment_customers, axis=1)
        return rfm_df
    
    def create_time_based_features(self, transaction_df):
        """Create time-based features from transaction data."""
        if 'InvoiceDate' not in transaction_df.columns:
            return transaction_df
        
        df = transaction_df.copy()
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        
        # Extract time components
        df['Year'] = df['InvoiceDate'].dt.year
        df['Month'] = df['InvoiceDate'].dt.month
        df['Quarter'] = df['InvoiceDate'].dt.quarter
        df['DayOfWeek'] = df['InvoiceDate'].dt.dayofweek
        df['DayOfMonth'] = df['InvoiceDate'].dt.day
        df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
        
        # Create cyclical features
        df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
        df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
        df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
        df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)
        
        return df
    
    def create_product_features(self, transaction_df):
        """Create product-related features if product information is available."""
        product_features = []
        
        if 'CustomerID' not in transaction_df.columns:
            return transaction_df
        
        df = transaction_df.copy()
        
        # Product diversity (if StockCode exists)
        if 'StockCode' in df.columns:
            product_diversity = df.groupby('CustomerID')['StockCode'].nunique().reset_index()
            product_diversity.columns = ['CustomerID', 'ProductDiversity']
            product_features.append(product_diversity)
        
        # Quantity patterns
        if 'Quantity' in df.columns:
            quantity_stats = df.groupby('CustomerID')['Quantity'].agg([
                'mean', 'std', 'min', 'max'
            ]).reset_index()
            quantity_stats.columns = ['CustomerID', 'AvgQuantity', 'StdQuantity', 'MinQuantity', 'MaxQuantity']
            quantity_stats['StdQuantity'] = quantity_stats['StdQuantity'].fillna(0)
            product_features.append(quantity_stats)
        
        # Merge all product features
        result_df = df.copy()
        for feature_df in product_features:
            result_df = result_df.merge(feature_df, on='CustomerID', how='left')
        
        return result_df
    
    def create_cohort_features(self, transaction_df):
        """Create cohort analysis features."""
        if 'CustomerID' not in transaction_df.columns or 'InvoiceDate' not in transaction_df.columns:
            return transaction_df
        
        df = transaction_df.copy()
        df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
        
        # Define cohorts based on first purchase month
        cohort_table = df.groupby('CustomerID')['InvoiceDate'].min().reset_index()
        cohort_table['CohortMonth'] = cohort_table['InvoiceDate'].dt.to_period('M')
        
        df = df.merge(cohort_table[['CustomerID', 'CohortMonth']], on='CustomerID')
        
        # Calculate cohort age (months since first purchase)
        df['InvoiceMonth'] = df['InvoiceDate'].dt.to_period('M')
        df['CohortAge'] = (df['InvoiceMonth'] - df['CohortMonth']).apply(attrgetter('n'))
        
        return df

def attrgetter(attr):
    """Helper function for cohort age calculation."""
    def getter(obj):
        return getattr(obj, attr)
    return getter
