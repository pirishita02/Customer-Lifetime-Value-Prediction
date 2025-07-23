import streamlit as st
import pandas as pd
import numpy as np
!pip install plotly
import plotly
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Import utility modules
from utils.data_processing import DataProcessor
from utils.feature_engineering import FeatureEngineer
from utils.models import MLModels
from utils.visualizations import Visualizer

def clean_dataframe_for_display(df):
    """Clean dataframe for Arrow compatibility in Streamlit."""
    df_clean = df.copy()
    
    # Convert all object columns to string to avoid mixed types
    for col in df_clean.columns:
        if df_clean[col].dtype == 'object':
            df_clean[col] = df_clean[col].astype(str)
        # Convert nullable integer columns to regular int
        elif hasattr(df_clean[col].dtype, 'name') and 'Int' in str(df_clean[col].dtype):
            df_clean[col] = df_clean[col].fillna(0).astype(int)
    
    return df_clean

# Page configuration
st.set_page_config(
    page_title="Customer Lifetime Value Prediction",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'predictions' not in st.session_state:
    st.session_state.predictions = None

# Title and description
st.title("💰 Customer Lifetime Value Prediction")
st.markdown("""
**Predict and analyze Customer Lifetime Value (CLV) using advanced machine learning models**

This application helps businesses understand and predict the total value a customer will bring over their entire relationship with the company.
""")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Choose a page:",
    ["📊 Data Upload & Overview", "🔍 Exploratory Data Analysis", "⚙️ Data Preprocessing", 
     "🤖 Model Training", "🎯 Predictions", "📚 About CLV"]
)

# Data Upload and Overview Page
if page == "📊 Data Upload & Overview":
    st.header("📊 Data Upload & Overview")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Upload Customer Data")
        uploaded_file = st.file_uploader(
            "Choose a CSV or Excel file",
            type=['csv', 'xlsx', 'xls'],
            help="Upload customer transaction data with columns like Customer ID, Purchase Amount, Purchase Date, etc."
        )
        
        if uploaded_file is not None:
            try:
                # Load data
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.session_state.data = df
                st.success(f"✅ Successfully loaded {len(df)} records!")
                
                # Display basic info
                st.subheader("Dataset Overview")
                col1_info, col2_info, col3_info = st.columns(3)
                
                with col1_info:
                    st.metric("Total Records", len(df))
                with col2_info:
                    st.metric("Total Columns", len(df.columns))
                with col3_info:
                    st.metric("Missing Values", df.isnull().sum().sum())
                
                # Display first few rows
                st.subheader("Data Preview")
                st.dataframe(clean_dataframe_for_display(df.head(10)), use_container_width=True)
                
                # Column information
                st.subheader("Column Information")
                col_info = pd.DataFrame({
                    'Column': df.columns,
                    'Data Type': df.dtypes.astype(str),
                    'Non-Null Count': df.count(),
                    'Null Count': df.isnull().sum(),
                    'Null Percentage': (df.isnull().sum() / len(df) * 100).round(2)
                })
                st.dataframe(col_info, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
    
    with col2:
        st.subheader("Sample Data")
        if st.button("📥 Load Sample Dataset", type="primary"):
            try:
                sample_df = pd.read_csv('sample_data/customer_sample.csv')
                st.session_state.data = sample_df
                st.success("✅ Sample data loaded!")
                st.rerun()
            except:
                st.error("❌ Sample data not available")
        
        st.subheader("Expected Data Format")
        st.markdown("""
        **Required/Recommended columns:**
        - `CustomerID`: Unique customer identifier
        - `InvoiceDate`: Transaction date
        - `Quantity`: Number of items purchased
        - `UnitPrice`: Price per unit
        - `TotalAmount`: Total purchase amount
        
        **Optional columns:**
        - `Country`: Customer location
        - `StockCode`: Product identifier
        - `Description`: Product description
        """)

# Exploratory Data Analysis Page
elif page == "🔍 Exploratory Data Analysis":
    st.header("🔍 Exploratory Data Analysis")
    
    if st.session_state.data is None:
        st.warning("⚠️ Please upload data first!")
    else:
        df = st.session_state.data
        visualizer = Visualizer()
        
        # EDA tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Statistical Summary", "📊 Distributions", "🔗 Relationships", "🗓️ Time Series"])
        
        with tab1:
            st.subheader("Statistical Summary")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Numerical Columns Summary**")
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    st.dataframe(df[numeric_cols].describe(), use_container_width=True)
                else:
                    st.info("No numerical columns found")
            
            with col2:
                st.markdown("**Categorical Columns Summary**")
                cat_cols = df.select_dtypes(include=['object']).columns.tolist()
                if cat_cols:
                    cat_summary = pd.DataFrame({
                        'Column': cat_cols,
                        'Unique Values': [df[col].nunique() for col in cat_cols],
                        'Most Frequent': [str(df[col].mode().iloc[0]) if len(df[col].mode()) > 0 else 'N/A' for col in cat_cols]
                    })
                    st.dataframe(cat_summary, use_container_width=True)
                else:
                    st.info("No categorical columns found")
        
        with tab2:
            st.subheader("Data Distributions")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                selected_col = st.selectbox("Select column for distribution analysis:", numeric_cols)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_hist = px.histogram(df, x=selected_col, title=f"Distribution of {selected_col}")
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                with col2:
                    fig_box = px.box(df, y=selected_col, title=f"Box Plot of {selected_col}")
                    st.plotly_chart(fig_box, use_container_width=True)
                
                # Outlier detection
                Q1 = df[selected_col].quantile(0.25)
                Q3 = df[selected_col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = df[(df[selected_col] < lower_bound) | (df[selected_col] > upper_bound)]
                
                st.info(f"📊 **{selected_col} Statistics:**\n"
                       f"- Mean: {df[selected_col].mean():.2f}\n"
                       f"- Median: {df[selected_col].median():.2f}\n"
                       f"- Std: {df[selected_col].std():.2f}\n"
                       f"- Outliers: {len(outliers)} ({len(outliers)/len(df)*100:.1f}%)")
        
        with tab3:
            st.subheader("Feature Relationships")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 2:
                # Correlation matrix
                st.markdown("**Correlation Matrix**")
                corr_matrix = df[numeric_cols].corr()
                fig_corr = px.imshow(corr_matrix, 
                                   title="Feature Correlation Matrix",
                                   color_continuous_scale='RdBu_r')
                st.plotly_chart(fig_corr, use_container_width=True)
                
                # Scatter plots
                col1, col2 = st.columns(2)
                with col1:
                    x_col = st.selectbox("Select X-axis:", numeric_cols, key='x_scatter')
                with col2:
                    y_col = st.selectbox("Select Y-axis:", 
                                       [col for col in numeric_cols if col != x_col], 
                                       key='y_scatter')
                
                fig_scatter = px.scatter(df, x=x_col, y=y_col, 
                                       title=f"{x_col} vs {y_col}")
                st.plotly_chart(fig_scatter, use_container_width=True)
        
        with tab4:
            st.subheader("Time Series Analysis")
            
            # Try to identify date columns
            date_cols = []
            for col in df.columns:
                if df[col].dtype == 'object':
                    try:
                        pd.to_datetime(df[col].iloc[0])
                        date_cols.append(col)
                    except:
                        continue
            
            if date_cols:
                selected_date_col = st.selectbox("Select date column:", date_cols)
                
                try:
                    df_temp = df.copy()
                    df_temp[selected_date_col] = pd.to_datetime(df_temp[selected_date_col])
                    
                    # Time series aggregation
                    agg_method = st.selectbox("Aggregation method:", 
                                            ["Daily", "Weekly", "Monthly"])
                    
                    if agg_method == "Daily":
                        freq = 'D'
                    elif agg_method == "Weekly":
                        freq = 'W'
                    else:
                        freq = 'M'
                    
                    # Group by time period
                    time_series = df_temp.set_index(selected_date_col).resample(freq).size()
                    
                    fig_ts = px.line(x=time_series.index, y=time_series.values,
                                   title=f"{agg_method} Transaction Count",
                                   labels={'x': 'Date', 'y': 'Count'})
                    st.plotly_chart(fig_ts, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error processing time series: {str(e)}")
            else:
                st.info("No date columns detected in the dataset")

# Data Preprocessing Page
elif page == "⚙️ Data Preprocessing":
    st.header("⚙️ Data Preprocessing")
    
    if st.session_state.data is None:
        st.warning("⚠️ Please upload data first!")
    else:
        df = st.session_state.data
        processor = DataProcessor()
        feature_engineer = FeatureEngineer()
        
        st.subheader("Feature Engineering & Data Processing")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("**Configure Processing Options**")
            
            # Column mapping
            st.markdown("**Map your columns to required fields:**")
            available_cols = df.columns.tolist()
            
            customer_id_col = st.selectbox("Customer ID column:", 
                                         ['None'] + available_cols,
                                         index=1 if len(available_cols) > 0 else 0)
            
            invoice_date_col = st.selectbox("Invoice Date column:", 
                                          ['None'] + available_cols,
                                          index=2 if len(available_cols) > 1 else 0)
            
            quantity_col = st.selectbox("Quantity column:", 
                                      ['None'] + available_cols,
                                      index=3 if len(available_cols) > 2 else 0)
            
            unit_price_col = st.selectbox("Unit Price column:", 
                                        ['None'] + available_cols,
                                        index=4 if len(available_cols) > 3 else 0)
            
            # Processing options
            st.markdown("**Processing Options:**")
            handle_missing = st.checkbox("Handle missing values", value=True)
            create_rfm = st.checkbox("Create RFM features", value=True)
            scale_features = st.checkbox("Scale numerical features", value=True)
            
        with col2:
            st.markdown("**Processing Information**")
            st.info("""
            **RFM Analysis:**
            - **Recency**: Days since last purchase
            - **Frequency**: Number of purchases
            - **Monetary**: Total amount spent
            
            **Additional Features:**
            - Customer tenure
            - Average order value
            - Purchase intervals
            """)
        
        if st.button("🚀 Process Data", type="primary"):
            with st.spinner("Processing data..."):
                try:
                    # Validate column mappings
                    if customer_id_col == 'None':
                        st.error("❌ Customer ID column is required!")
                        st.stop()
                    
                    processed_df = df.copy()
                    
                    # Rename columns to standard names (avoid conflicts)
                    column_mapping = {}
                    if customer_id_col != 'None' and customer_id_col != 'CustomerID':
                        column_mapping[customer_id_col] = 'CustomerID'
                    if invoice_date_col != 'None' and invoice_date_col != 'InvoiceDate':
                        column_mapping[invoice_date_col] = 'InvoiceDate'
                    if quantity_col != 'None' and quantity_col != 'Quantity':
                        column_mapping[quantity_col] = 'Quantity'
                    if unit_price_col != 'None' and unit_price_col != 'UnitPrice':
                        column_mapping[unit_price_col] = 'UnitPrice'
                    
                    if column_mapping:
                        processed_df = processed_df.rename(columns=column_mapping)
                    
                    # Create TotalAmount if not exists
                    if 'Quantity' in processed_df.columns and 'UnitPrice' in processed_df.columns:
                        processed_df['TotalAmount'] = processed_df['Quantity'] * processed_df['UnitPrice']
                    
                    # Handle missing values
                    if handle_missing:
                        processed_df = processor.handle_missing_values(processed_df)
                    
                    # Create RFM features
                    if create_rfm and 'CustomerID' in processed_df.columns:
                        if 'InvoiceDate' in processed_df.columns:
                            processed_df['InvoiceDate'] = pd.to_datetime(processed_df['InvoiceDate'])
                        
                        st.info("Creating RFM features...")
                        rfm_df = feature_engineer.create_rfm_features(processed_df)
                        
                        st.info("Calculating CLV target variable...")
                        rfm_df = feature_engineer.calculate_clv(rfm_df)
                        
                        # Validate CLV data before proceeding
                        numeric_cols = rfm_df.select_dtypes(include=[np.number]).columns
                        for col in numeric_cols:
                            if np.isinf(rfm_df[col]).any():
                                st.warning(f"⚠️ Infinite values detected in {col}, cleaning data...")
                                rfm_df[col] = rfm_df[col].replace([np.inf, -np.inf], rfm_df[col].median())
                            
                            if (rfm_df[col] > 1e10).any():
                                st.warning(f"⚠️ Extremely large values detected in {col}, capping values...")
                                rfm_df[col] = np.clip(rfm_df[col], -1e6, 1e6)
                        
                        processed_df = rfm_df
                    
                    # Scale features
                    if scale_features:
                        st.info("Scaling features...")
                        # Additional safety check before scaling
                        numeric_cols_to_check = processed_df.select_dtypes(include=[np.number]).columns
                        has_inf = any(np.isinf(processed_df[col]).any() for col in numeric_cols_to_check)
                        has_large = any((processed_df[col].abs() > 1e8).any() for col in numeric_cols_to_check)
                        
                        if has_inf or has_large:
                            st.warning("⚠️ Data contains extreme values, applying additional cleaning...")
                            for col in numeric_cols_to_check:
                                processed_df[col] = processed_df[col].replace([np.inf, -np.inf], np.nan)
                                processed_df[col] = processed_df[col].fillna(processed_df[col].median())
                                processed_df[col] = np.clip(processed_df[col], -1e6, 1e6)
                        
                        processed_df = processor.scale_features(processed_df)
                    
                    st.session_state.processed_data = processed_df
                    st.success("✅ Data processing completed!")
                    
                    # Display processed data info
                    st.subheader("Processed Data Overview")
                    
                    col1_info, col2_info, col3_info = st.columns(3)
                    with col1_info:
                        st.metric("Processed Records", len(processed_df))
                    with col2_info:
                        st.metric("Features Created", len(processed_df.columns))
                    with col3_info:
                        st.metric("Missing Values", processed_df.isnull().sum().sum())
                    
                    st.dataframe(clean_dataframe_for_display(processed_df.head()), use_container_width=True)
                    
                    # Download processed data
                    csv = processed_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Processed Data",
                        data=csv,
                        file_name="processed_customer_data.csv",
                        mime="text/csv"
                    )
                    
                except Exception as e:
                    st.error(f"❌ Error processing data: {str(e)}")

# Model Training Page
elif page == "🤖 Model Training":
    st.header("🤖 Model Training")
    
    if st.session_state.processed_data is None:
        st.warning("⚠️ Please process data first!")
    else:
        df = st.session_state.processed_data
        ml_models = MLModels()
        
        st.subheader("Train Machine Learning Models")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Target variable selection
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            target_col = st.selectbox("Select target variable (CLV):", 
                                    numeric_cols,
                                    index=len(numeric_cols)-1 if 'CLV' in numeric_cols[-1:] else 0)
            
            # Feature selection
            feature_cols = [col for col in numeric_cols if col != target_col]
            selected_features = st.multiselect("Select features:", 
                                             feature_cols,
                                             default=feature_cols[:10] if len(feature_cols) > 10 else feature_cols)
            
            # Model selection
            model_types = st.multiselect("Select models to train:",
                                       ["Random Forest", "XGBoost", "Gradient Boosting"],
                                       default=["Random Forest", "XGBoost"])
            
            # Train/test split
            test_size = st.slider("Test set size:", 0.1, 0.4, 0.2, 0.05)
            
        with col2:
            st.markdown("**Model Information**")
            st.info("""
            **Random Forest:**
            - Ensemble of decision trees
            - Good for non-linear relationships
            - Provides feature importance
            
            **XGBoost:**
            - Gradient boosting algorithm
            - Often achieves high accuracy
            - Handles missing values well
            
            **Metrics:**
            - MAE: Mean Absolute Error
            - RMSE: Root Mean Square Error
            - R²: Coefficient of Determination
            """)
        
        if st.button("🚀 Train Models", type="primary"):
            if not selected_features:
                st.error("❌ Please select at least one feature!")
            elif not model_types:
                st.error("❌ Please select at least one model!")
            else:
                with st.spinner("Training models..."):
                    try:
                        # Prepare data
                        X = df[selected_features]
                        y = df[target_col]
                        
                        # Remove any remaining NaN values
                        mask = ~(X.isnull().any(axis=1) | y.isnull())
                        X = X[mask]
                        y = y[mask]
                        
                        if len(X) == 0:
                            st.error("❌ No valid data remaining after cleaning!")
                            st.stop()
                        
                        # Check data size before training
                        st.info(f"Training with {len(X)} samples and {len(selected_features)} features")
                        
                        if len(X) < 5:
                            st.error(f"❌ Need at least 5 unique customers for model training, but only got {len(X)}. "
                                   f"Please upload a dataset with more customers or check your data preprocessing.")
                            st.stop()
                        
                        # Train models
                        results = ml_models.train_models(X, y, model_types, test_size)
                        st.session_state.models = results
                        
                        st.success("✅ Models trained successfully!")
                        
                        # Display results
                        st.subheader("Model Performance")
                        
                        # Create results DataFrame
                        results_df = pd.DataFrame({
                            'Model': results.keys(),
                            'MAE': [results[model]['metrics']['mae'] for model in results.keys()],
                            'RMSE': [results[model]['metrics']['rmse'] for model in results.keys()],
                            'R²': [results[model]['metrics']['r2'] for model in results.keys()]
                        })
                        
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Visualize model comparison
                        fig_metrics = make_subplots(
                            rows=1, cols=3,
                            subplot_titles=('Mean Absolute Error', 'Root Mean Square Error', 'R² Score')
                        )
                        
                        fig_metrics.add_trace(
                            go.Bar(x=results_df['Model'], y=results_df['MAE'], name='MAE'),
                            row=1, col=1
                        )
                        fig_metrics.add_trace(
                            go.Bar(x=results_df['Model'], y=results_df['RMSE'], name='RMSE'),
                            row=1, col=2
                        )
                        fig_metrics.add_trace(
                            go.Bar(x=results_df['Model'], y=results_df['R²'], name='R²'),
                            row=1, col=3
                        )
                        
                        fig_metrics.update_layout(title="Model Performance Comparison", showlegend=False)
                        st.plotly_chart(fig_metrics, use_container_width=True)
                        
                        # Feature importance
                        best_model = max(results.keys(), key=lambda x: results[x]['metrics']['r2'])
                        st.subheader(f"Feature Importance - {best_model}")
                        
                        if 'feature_importance' in results[best_model]:
                            importance_df = results[best_model]['feature_importance']
                            fig_importance = px.bar(importance_df, 
                                                  x='importance', 
                                                  y='feature',
                                                  orientation='h',
                                                  title=f"{best_model} Feature Importance")
                            st.plotly_chart(fig_importance, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"❌ Error training models: {str(e)}")

# Predictions Page
elif page == "🎯 Predictions":
    st.header("🎯 Customer Lifetime Value Predictions")
    
    if not st.session_state.models:
        st.warning("⚠️ Please train models first!")
    else:
        df = st.session_state.processed_data
        models = st.session_state.models
        
        tab1, tab2 = st.tabs(["🔮 Single Prediction", "📊 Batch Predictions"])
        
        with tab1:
            st.subheader("Predict CLV for a Single Customer")
            
            # Get feature names from trained models
            best_model_name = max(models.keys(), key=lambda x: models[x]['metrics']['r2'])
            feature_names = models[best_model_name]['feature_names']
            
            # Create input form
            col1, col2 = st.columns(2)
            
            input_data = {}
            
            for i, feature in enumerate(feature_names):
                col = col1 if i % 2 == 0 else col2
                with col:
                    # Get feature statistics for reasonable defaults
                    if feature in df.columns:
                        min_val = float(df[feature].min())
                        max_val = float(df[feature].max())
                        mean_val = float(df[feature].mean())
                        input_data[feature] = st.number_input(
                            f"{feature}:",
                            min_value=min_val,
                            max_value=max_val,
                            value=mean_val,
                            help=f"Range: {min_val:.2f} - {max_val:.2f}"
                        )
                    else:
                        input_data[feature] = st.number_input(f"{feature}:", value=0.0)
            
            # Model selection for prediction
            selected_model = st.selectbox("Select model for prediction:", 
                                        list(models.keys()))
            
            if st.button("🎯 Predict CLV", type="primary"):
                try:
                    # Prepare input
                    input_df = pd.DataFrame([input_data])
                    
                    # Make prediction
                    model = models[selected_model]['model']
                    prediction = model.predict(input_df)[0]
                    
                    # Calculate confidence interval if available
                    try:
                        if hasattr(model, 'predict'):
                            # For tree-based models, we can estimate uncertainty
                            std_error = models[selected_model]['metrics']['rmse']
                            confidence_lower = prediction - 1.96 * std_error
                            confidence_upper = prediction + 1.96 * std_error
                        else:
                            confidence_lower = prediction * 0.8
                            confidence_upper = prediction * 1.2
                    except:
                        confidence_lower = prediction * 0.8
                        confidence_upper = prediction * 1.2
                    
                    # Display results
                    st.success("✅ Prediction completed!")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Predicted CLV", f"${prediction:,.2f}")
                    with col2:
                        st.metric("Lower Bound (95%)", f"${confidence_lower:,.2f}")
                    with col3:
                        st.metric("Upper Bound (95%)", f"${confidence_upper:,.2f}")
                    
                    # Interpretation
                    st.subheader("Interpretation")
                    if prediction > df['CLV'].quantile(0.75) if 'CLV' in df.columns else prediction > 1000:
                        st.info("🌟 **High-Value Customer**: This customer is predicted to have a high lifetime value. Consider premium retention strategies.")
                    elif prediction > df['CLV'].quantile(0.5) if 'CLV' in df.columns else prediction > 500:
                        st.info("📈 **Medium-Value Customer**: This customer shows good potential. Focus on engagement and upselling.")
                    else:
                        st.info("📊 **Standard Customer**: Consider strategies to increase engagement and purchase frequency.")
                    
                except Exception as e:
                    st.error(f"❌ Error making prediction: {str(e)}")
        
        with tab2:
            st.subheader("Batch Predictions on Dataset")
            
            # Select model for batch predictions
            selected_model = st.selectbox("Select model for batch predictions:", 
                                        list(models.keys()),
                                        key='batch_model')
            
            if st.button("🚀 Generate Batch Predictions", type="primary"):
                try:
                    with st.spinner("Generating predictions..."):
                        # Get feature names and prepare data
                        feature_names = models[selected_model]['feature_names']
                        X = df[feature_names]
                        
                        # Make predictions
                        model = models[selected_model]['model']
                        predictions = model.predict(X)
                        
                        # Create results DataFrame
                        results_df = df.copy()
                        results_df['Predicted_CLV'] = predictions
                        
                        # Add prediction categories
                        results_df['CLV_Category'] = pd.cut(
                            predictions,
                            bins=3,
                            labels=['Low', 'Medium', 'High']
                        )
                        
                        st.session_state.predictions = results_df
                        st.success("✅ Batch predictions completed!")
                        
                        # Display summary
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Predictions", len(results_df))
                        with col2:
                            st.metric("Average Predicted CLV", f"${predictions.mean():,.2f}")
                        with col3:
                            st.metric("Total Predicted Revenue", f"${predictions.sum():,.2f}")
                        
                        # Display predictions
                        st.subheader("Prediction Results")
                        st.dataframe(results_df[['CustomerID', 'Predicted_CLV', 'CLV_Category']].head(20) if 'CustomerID' in results_df.columns else results_df[['Predicted_CLV', 'CLV_Category']].head(20), 
                                   use_container_width=True)
                        
                        # Visualizations
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # CLV distribution
                            fig_dist = px.histogram(results_df, x='Predicted_CLV',
                                                  title="Distribution of Predicted CLV")
                            st.plotly_chart(fig_dist, use_container_width=True)
                        
                        with col2:
                            # CLV categories
                            category_counts = results_df['CLV_Category'].value_counts()
                            fig_cat = px.pie(values=category_counts.values, 
                                           names=category_counts.index,
                                           title="Customer Value Categories")
                            st.plotly_chart(fig_cat, use_container_width=True)
                        
                        # Download results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Predictions",
                            data=csv,
                            file_name="clv_predictions.csv",
                            mime="text/csv"
                        )
                        
                except Exception as e:
                    st.error(f"❌ Error generating batch predictions: {str(e)}")

# About CLV Page
elif page == "📚 About CLV":
    st.header("📚 Understanding Customer Lifetime Value")
    
    st.markdown("""
    ## What is Customer Lifetime Value (CLV)?
    
    **Customer Lifetime Value (CLV)** is a prediction of the total revenue a business can expect from a single customer account throughout their business relationship. It's one of the most important metrics for understanding customer profitability and making informed business decisions.
    
    ### Why is CLV Important?
    
    1. **Customer Acquisition**: Helps determine how much to spend on acquiring new customers
    2. **Resource Allocation**: Identify which customers deserve more attention and resources
    3. **Marketing Strategy**: Tailor marketing efforts based on customer value segments
    4. **Product Development**: Understand what features/products drive higher lifetime value
    5. **Customer Retention**: Focus retention efforts on high-value customers
    
    ### How is CLV Calculated?
    
    There are several approaches to calculate CLV:
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### Traditional Formula
        ```
        CLV = (Average Purchase Value × Purchase Frequency × Customer Lifespan)
        ```
        
        #### RFM-Based Approach
        - **Recency**: How recently did the customer make a purchase?
        - **Frequency**: How often do they purchase?
        - **Monetary**: How much do they spend?
        """)
    
    with col2:
        st.markdown("""
        #### Machine Learning Approach
        Uses historical data to predict future behavior:
        - Transaction history
        - Customer demographics
        - Behavioral patterns
        - Seasonal trends
        """)
    
    st.markdown("""
    ### Key Features Used in This Application
    
    1. **Recency**: Days since the last purchase
    2. **Frequency**: Total number of purchases
    3. **Monetary Value**: Total amount spent
    4. **Customer Tenure**: Time since first purchase
    5. **Average Order Value**: Mean purchase amount
    6. **Purchase Intervals**: Time between purchases
    
    ### Machine Learning Models
    
    This application uses ensemble methods for CLV prediction:
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        #### Random Forest
        - **Pros**: Handles non-linear relationships, provides feature importance
        - **Cons**: Can overfit with small datasets
        - **Best for**: Complex customer behavior patterns
        """)
    
    with col2:
        st.markdown("""
        #### XGBoost
        - **Pros**: High accuracy, handles missing values
        - **Cons**: Requires parameter tuning
        - **Best for**: Large datasets with mixed data types
        """)
    
    with col3:
        st.markdown("""
        #### Gradient Boosting
        - **Pros**: Sequential learning, good generalization
        - **Cons**: Sensitive to outliers
        - **Best for**: Structured tabular data
        """)
    
    st.markdown("""
    ### Business Applications
    
    #### Customer Segmentation
    - **High CLV**: Premium customers (top 20%)
    - **Medium CLV**: Growth potential customers (next 30%)
    - **Low CLV**: Cost-conscious or new customers (remaining 50%)
    
    #### Marketing Strategies
    - **High CLV**: Personalized service, loyalty programs, premium offerings
    - **Medium CLV**: Upselling, cross-selling, engagement campaigns
    - **Low CLV**: Retention programs, value-based offers
    
    #### Resource Allocation
    - Allocate customer service resources based on CLV
    - Prioritize product development for high-CLV segments
    - Optimize marketing spend across customer tiers
    
    ### Best Practices
    
    1. **Regular Updates**: Recalculate CLV regularly as customer behavior changes
    2. **Segment-Specific Models**: Consider different models for different customer segments
    3. **External Factors**: Include seasonality, economic conditions, and market trends
    4. **Validation**: Continuously validate predictions against actual outcomes
    5. **Business Context**: Always interpret CLV in the context of your specific business model
    
    ### Getting Started
    
    To use this application effectively:
    
    1. **Prepare Your Data**: Ensure you have customer transaction data with dates, amounts, and customer IDs
    2. **Upload and Explore**: Use the EDA features to understand your customer patterns
    3. **Feature Engineering**: Let the app create RFM and other predictive features
    4. **Train Models**: Compare different algorithms to find the best fit
    5. **Make Predictions**: Use the trained models to predict CLV for new or existing customers
    6. **Take Action**: Apply insights to your business strategy
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Customer Lifetime Value Prediction App | Built with Streamlit</p>
    <p>💡 Use the sidebar to navigate between different sections</p>
</div>
""", unsafe_allow_html=True)
