import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

class Visualizer:
    """Create interactive visualizations for Customer Lifetime Value analysis."""
    
    def __init__(self):
        self.color_palette = px.colors.qualitative.Set3
    
    def plot_clv_distribution(self, df, clv_column='CLV', title="Customer Lifetime Value Distribution"):
        """Plot CLV distribution with statistics."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('CLV Distribution', 'CLV Box Plot', 'CLV by Segments', 'CLV Statistics'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"type": "table"}]]
        )
        
        # Histogram
        fig.add_trace(
            go.Histogram(x=df[clv_column], name="CLV Distribution", nbinsx=30),
            row=1, col=1
        )
        
        # Box plot
        fig.add_trace(
            go.Box(y=df[clv_column], name="CLV Box Plot"),
            row=1, col=2
        )
        
        # CLV by segments (if available)
        if 'CustomerSegment' in df.columns:
            segment_clv = df.groupby('CustomerSegment')[clv_column].mean().sort_values(ascending=False)
            fig.add_trace(
                go.Bar(x=segment_clv.index, y=segment_clv.values, name="Avg CLV by Segment"),
                row=2, col=1
            )
        
        # Statistics table
        stats = df[clv_column].describe()
        fig.add_trace(
            go.Table(
                header=dict(values=['Statistic', 'Value']),
                cells=dict(values=[
                    ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max'],
                    [f"{stats['count']:.0f}", f"${stats['mean']:.2f}", f"${stats['std']:.2f}",
                     f"${stats['min']:.2f}", f"${stats['25%']:.2f}", f"${stats['50%']:.2f}",
                     f"${stats['75%']:.2f}", f"${stats['max']:.2f}"]
                ])
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title=title,
            height=800,
            showlegend=False
        )
        
        return fig
    
    def plot_rfm_analysis(self, df):
        """Create comprehensive RFM analysis visualizations."""
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=('Recency Distribution', 'Frequency Distribution', 'Monetary Distribution',
                          'R vs F', 'R vs M', 'F vs M'),
            specs=[[{"secondary_y": False} for _ in range(3)],
                   [{"secondary_y": False} for _ in range(3)]]
        )
        
        # RFM distributions
        fig.add_trace(go.Histogram(x=df['Recency'], name="Recency", nbinsx=20), row=1, col=1)
        fig.add_trace(go.Histogram(x=df['Frequency'], name="Frequency", nbinsx=20), row=1, col=2)
        fig.add_trace(go.Histogram(x=df['Monetary'], name="Monetary", nbinsx=20), row=1, col=3)
        
        # RFM scatter plots
        fig.add_trace(go.Scatter(x=df['Recency'], y=df['Frequency'], mode='markers', name="R vs F"), row=2, col=1)
        fig.add_trace(go.Scatter(x=df['Recency'], y=df['Monetary'], mode='markers', name="R vs M"), row=2, col=2)
        fig.add_trace(go.Scatter(x=df['Frequency'], y=df['Monetary'], mode='markers', name="F vs M"), row=2, col=3)
        
        fig.update_layout(
            title="RFM Analysis Dashboard",
            height=800,
            showlegend=False
        )
        
        return fig
    
    def plot_customer_segments(self, df):
        """Visualize customer segments based on RFM scores."""
        if 'CustomerSegment' not in df.columns:
            return None
        
        segment_summary = df.groupby('CustomerSegment').agg({
            'Recency': 'mean',
            'Frequency': 'mean',
            'Monetary': 'mean',
            'CLV': 'mean' if 'CLV' in df.columns else 'count'
        }).reset_index()
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Segment Distribution', 'Avg Recency by Segment', 
                          'Avg Frequency by Segment', 'Avg Monetary by Segment'),
            specs=[[{"type": "pie"}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Pie chart of segment distribution
        segment_counts = df['CustomerSegment'].value_counts()
        fig.add_trace(
            go.Pie(labels=segment_counts.index, values=segment_counts.values, name="Segments"),
            row=1, col=1
        )
        
        # Bar charts for each RFM dimension
        fig.add_trace(
            go.Bar(x=segment_summary['CustomerSegment'], y=segment_summary['Recency'], name="Recency"),
            row=1, col=2
        )
        fig.add_trace(
            go.Bar(x=segment_summary['CustomerSegment'], y=segment_summary['Frequency'], name="Frequency"),
            row=2, col=1
        )
        fig.add_trace(
            go.Bar(x=segment_summary['CustomerSegment'], y=segment_summary['Monetary'], name="Monetary"),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Customer Segmentation Analysis",
            height=800,
            showlegend=False
        )
        
        return fig
    
    def plot_feature_importance(self, feature_importance_df, title="Feature Importance"):
        """Plot feature importance from ML models."""
        fig = px.bar(
            feature_importance_df.head(15),  # Top 15 features
            x='importance',
            y='feature',
            orientation='h',
            title=title
        )
        
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            height=600
        )
        
        return fig
    
    def plot_model_performance(self, results_df):
        """Plot model performance comparison."""
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=('Mean Absolute Error', 'Root Mean Square Error', 'R² Score')
        )
        
        fig.add_trace(
            go.Bar(x=results_df['Model'], y=results_df['MAE'], name='MAE'),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(x=results_df['Model'], y=results_df['RMSE'], name='RMSE'),
            row=1, col=2
        )
        fig.add_trace(
            go.Bar(x=results_df['Model'], y=results_df['R²'], name='R²'),
            row=1, col=3
        )
        
        fig.update_layout(
            title="Model Performance Comparison",
            height=400,
            showlegend=False
        )
        
        return fig
    
    def plot_predictions_vs_actual(self, y_true, y_pred, model_name="Model"):
        """Plot predictions vs actual values."""
        fig = go.Figure()
        
        # Scatter plot
        fig.add_trace(go.Scatter(
            x=y_true,
            y=y_pred,
            mode='markers',
            name='Predictions',
            opacity=0.6
        ))
        
        # Perfect prediction line
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title=f"{model_name}: Predictions vs Actual Values",
            xaxis_title="Actual CLV",
            yaxis_title="Predicted CLV",
            height=500
        )
        
        return fig
    
    def plot_residuals_analysis(self, y_true, y_pred, model_name="Model"):
        """Plot residuals analysis."""
        residuals = y_true - y_pred
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Residuals vs Predicted', 'Residuals Distribution')
        )
        
        # Residuals vs predicted
        fig.add_trace(
            go.Scatter(x=y_pred, y=residuals, mode='markers', name='Residuals'),
            row=1, col=1
        )
        
        # Add horizontal line at y=0
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=1)
        
        # Residuals histogram
        fig.add_trace(
            go.Histogram(x=residuals, name='Residuals Distribution', nbinsx=30),
            row=1, col=2
        )
        
        fig.update_layout(
            title=f"{model_name}: Residuals Analysis",
            height=400,
            showlegend=False
        )
        
        return fig
    
    def plot_clv_trends(self, df, date_col='InvoiceDate', clv_col='CLV'):
        """Plot CLV trends over time."""
        if date_col not in df.columns:
            return None
        
        df_temp = df.copy()
        df_temp[date_col] = pd.to_datetime(df_temp[date_col])
        
        # Aggregate by month
        monthly_clv = df_temp.groupby(df_temp[date_col].dt.to_period('M'))[clv_col].mean().reset_index()
        monthly_clv[date_col] = monthly_clv[date_col].astype(str)
        
        fig = px.line(
            monthly_clv,
            x=date_col,
            y=clv_col,
            title="Average CLV Trends Over Time"
        )
        
        fig.update_layout(
            xaxis_title="Month",
            yaxis_title="Average CLV",
            height=400
        )
        
        return fig
    
    def plot_cohort_analysis(self, df):
        """Create cohort analysis visualization."""
        if not all(col in df.columns for col in ['CustomerID', 'InvoiceDate']):
            return None
        
        df_temp = df.copy()
        df_temp['InvoiceDate'] = pd.to_datetime(df_temp['InvoiceDate'])
        
        # Create cohort table
        cohort_table = df_temp.groupby('CustomerID')['InvoiceDate'].min().reset_index()
        cohort_table['CohortMonth'] = cohort_table['InvoiceDate'].dt.to_period('M')
        
        df_temp = df_temp.merge(cohort_table[['CustomerID', 'CohortMonth']], on='CustomerID')
        df_temp['InvoiceMonth'] = df_temp['InvoiceDate'].dt.to_period('M')
        
        # Calculate retention rates
        cohort_data = df_temp.groupby(['CohortMonth', 'InvoiceMonth'])['CustomerID'].nunique().reset_index()
        cohort_counts = cohort_table['CohortMonth'].value_counts().sort_index()
        
        cohort_table_pivot = cohort_data.pivot(index='CohortMonth', 
                                             columns='InvoiceMonth', 
                                             values='CustomerID')
        
        # Calculate retention rates
        cohort_sizes = cohort_table.groupby('CohortMonth')['CustomerID'].nunique()
        retention_table = cohort_table_pivot.divide(cohort_sizes, axis=0)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=retention_table.values,
            x=[str(col) for col in retention_table.columns],
            y=[str(idx) for idx in retention_table.index],
            colorscale='Blues',
            showscale=True
        ))
        
        fig.update_layout(
            title="Cohort Analysis - Customer Retention Rates",
            xaxis_title="Invoice Month",
            yaxis_title="Cohort Month",
            height=600
        )
        
        return fig
    
    def create_clv_dashboard(self, df):
        """Create a comprehensive CLV dashboard."""
        if 'CLV' not in df.columns:
            return None
        
        # Create multiple visualizations
        figs = []
        
        # CLV Distribution
        figs.append(self.plot_clv_distribution(df))
        
        # RFM Analysis
        if all(col in df.columns for col in ['Recency', 'Frequency', 'Monetary']):
            figs.append(self.plot_rfm_analysis(df))
        
        # Customer Segments
        if 'CustomerSegment' in df.columns:
            figs.append(self.plot_customer_segments(df))
        
        # CLV Trends
        if 'InvoiceDate' in df.columns:
            figs.append(self.plot_clv_trends(df))
        
        return figs
