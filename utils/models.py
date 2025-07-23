import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

class MLModels:
    """Machine Learning models for Customer Lifetime Value prediction."""
    
    def __init__(self):
        self.models = {}
        self.results = {}
    
    def train_models(self, X, y, model_types, test_size=0.2, random_state=42):
        """Train multiple ML models and return performance metrics."""
        # Check if we have enough samples
        n_samples = len(X)
        if n_samples < 5:
            raise ValueError(f"Need at least 5 samples for training, but only got {n_samples}. "
                           f"Please ensure your dataset has multiple unique customers.")
        
        # Adjust test_size and CV folds based on sample size
        if n_samples < 10:
            test_size = 0.1  # Use smaller test size for small datasets
            cv_folds = min(3, n_samples)  # Use fewer CV folds
        elif n_samples < 20:
            cv_folds = min(3, n_samples)
        else:
            cv_folds = 5
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        results = {}
        
        for model_type in model_types:
            print(f"Training {model_type}...")
            
            # Adjust model parameters for small datasets
            if n_samples < 20:
                # Use simpler models for small datasets
                if model_type == "Random Forest":
                    model = RandomForestRegressor(
                        n_estimators=min(50, n_samples * 5),
                        max_depth=min(5, int(np.log2(n_samples)) + 1),
                        min_samples_split=2,
                        min_samples_leaf=1,
                        random_state=random_state,
                        n_jobs=-1
                    )
                elif model_type == "XGBoost":
                    model = xgb.XGBRegressor(
                        n_estimators=min(50, n_samples * 5),
                        max_depth=min(3, int(np.log2(n_samples)) + 1),
                        learning_rate=0.1,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=random_state,
                        n_jobs=-1
                    )
                elif model_type == "Gradient Boosting":
                    model = GradientBoostingRegressor(
                        n_estimators=min(50, n_samples * 5),
                        max_depth=min(3, int(np.log2(n_samples)) + 1),
                        learning_rate=0.1,
                        subsample=0.8,
                        random_state=random_state
                    )
                else:
                    continue
            else:
                # Use full model parameters for larger datasets
                if model_type == "Random Forest":
                    model = RandomForestRegressor(
                        n_estimators=100,
                        max_depth=10,
                        min_samples_split=5,
                        min_samples_leaf=2,
                        random_state=random_state,
                        n_jobs=-1
                    )
                elif model_type == "XGBoost":
                    model = xgb.XGBRegressor(
                        n_estimators=100,
                        max_depth=6,
                        learning_rate=0.1,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=random_state,
                        n_jobs=-1
                    )
                elif model_type == "Gradient Boosting":
                    model = GradientBoostingRegressor(
                        n_estimators=100,
                        max_depth=6,
                        learning_rate=0.1,
                        subsample=0.8,
                        random_state=random_state
                    )
                else:
                    continue
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            # Cross-validation score (adjust CV folds)
            if len(X_train) >= cv_folds:
                cv_scores = cross_val_score(model, X_train, y_train, cv=cv_folds, scoring='r2')
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
            else:
                # Skip CV if not enough samples
                cv_mean = r2  # Use test R2 as proxy
                cv_std = 0.0
                print(f"Warning: Skipping cross-validation for {model_type} due to insufficient samples")
            
            # Feature importance (if available)
            feature_importance = None
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': X.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                feature_importance = importance_df
            
            # Store results
            results[model_type] = {
                'model': model,
                'metrics': {
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2,
                    'cv_mean': cv_mean,
                    'cv_std': cv_std
                },
                'feature_importance': feature_importance,
                'predictions': y_pred,
                'y_test': y_test,
                'feature_names': X.columns.tolist(),
                'n_samples': n_samples,
                'cv_folds': cv_folds
            }
        
        self.results = results
        return results
    
    def predict_single(self, model, features):
        """Make prediction for a single sample."""
        if isinstance(features, dict):
            features_df = pd.DataFrame([features])
        else:
            features_df = features
        
        prediction = model.predict(features_df)
        return prediction[0] if len(prediction) == 1 else prediction
    
    def predict_with_confidence(self, model_name, features, confidence_level=0.95):
        """Make prediction with confidence interval."""
        if model_name not in self.results:
            raise ValueError(f"Model {model_name} not found. Available models: {list(self.results.keys())}")
        
        model = self.results[model_name]['model']
        prediction = self.predict_single(model, features)
        
        # Estimate confidence interval using residuals
        y_test = self.results[model_name]['y_test']
        y_pred = self.results[model_name]['predictions']
        residuals = y_test - y_pred
        residual_std = np.std(residuals)
        
        # Calculate confidence interval
        z_score = 1.96 if confidence_level == 0.95 else 2.576  # 99% confidence
        margin_of_error = z_score * residual_std
        
        confidence_lower = prediction - margin_of_error
        confidence_upper = prediction + margin_of_error
        
        return {
            'prediction': prediction,
            'confidence_lower': confidence_lower,
            'confidence_upper': confidence_upper,
            'confidence_level': confidence_level
        }
    
    def get_model_comparison(self):
        """Get comparison of all trained models."""
        if not self.results:
            return None
        
        comparison_data = []
        for model_name, result in self.results.items():
            metrics = result['metrics']
            comparison_data.append({
                'Model': model_name,
                'MAE': metrics['mae'],
                'RMSE': metrics['rmse'],
                'R²': metrics['r2'],
                'CV Mean R²': metrics['cv_mean'],
                'CV Std R²': metrics['cv_std']
            })
        
        return pd.DataFrame(comparison_data)
    
    def get_best_model(self, metric='r2'):
        """Get the best performing model based on specified metric."""
        if not self.results:
            return None
        
        best_model_name = max(self.results.keys(), 
                            key=lambda x: self.results[x]['metrics'][metric])
        return best_model_name, self.results[best_model_name]
    
    def analyze_predictions(self, model_name):
        """Analyze prediction quality and residuals."""
        if model_name not in self.results:
            return None
        
        y_test = self.results[model_name]['y_test']
        y_pred = self.results[model_name]['predictions']
        
        residuals = y_test - y_pred
        
        analysis = {
            'residuals': residuals,
            'residual_mean': np.mean(residuals),
            'residual_std': np.std(residuals),
            'residual_min': np.min(residuals),
            'residual_max': np.max(residuals),
            'prediction_accuracy': {
                'within_10_percent': np.sum(np.abs(residuals) <= 0.1 * y_test) / len(y_test),
                'within_20_percent': np.sum(np.abs(residuals) <= 0.2 * y_test) / len(y_test),
                'within_50_percent': np.sum(np.abs(residuals) <= 0.5 * y_test) / len(y_test)
            }
        }
        
        return analysis
    
    def hyperparameter_tuning(self, model_type, X, y, param_grid=None):
        """Perform hyperparameter tuning for specified model."""
        from sklearn.model_selection import GridSearchCV
        
        if model_type == "Random Forest":
            if param_grid is None:
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, 15, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            model = RandomForestRegressor(random_state=42)
            
        elif model_type == "XGBoost":
            if param_grid is None:
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0]
                }
            model = xgb.XGBRegressor(random_state=42)
        
        else:
            raise ValueError(f"Hyperparameter tuning not implemented for {model_type}")
        
        # Perform grid search
        grid_search = GridSearchCV(
            model, param_grid, cv=5, scoring='r2', n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X, y)
        
        return {
            'best_model': grid_search.best_estimator_,
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
    
    def ensemble_predict(self, features, weights=None):
        """Make ensemble prediction using all trained models."""
        if not self.results:
            raise ValueError("No models trained yet")
        
        predictions = []
        model_names = []
        
        for model_name, result in self.results.items():
            model = result['model']
            pred = self.predict_single(model, features)
            predictions.append(pred)
            model_names.append(model_name)
        
        predictions = np.array(predictions)
        
        if weights is None:
            # Use R² scores as weights
            weights = np.array([self.results[name]['metrics']['r2'] for name in model_names])
            weights = weights / np.sum(weights)  # Normalize
        
        ensemble_prediction = np.average(predictions, weights=weights)
        
        return {
            'ensemble_prediction': ensemble_prediction,
            'individual_predictions': dict(zip(model_names, predictions)),
            'weights': dict(zip(model_names, weights))
        }
    
    def save_models(self, filepath_prefix):
        """Save trained models to disk."""
        import joblib
        
        saved_models = {}
        for model_name, result in self.results.items():
            filepath = f"{filepath_prefix}_{model_name.lower().replace(' ', '_')}.joblib"
            joblib.dump(result['model'], filepath)
            saved_models[model_name] = filepath
        
        return saved_models
    
    def load_model(self, filepath, model_name):
        """Load a trained model from disk."""
        import joblib
        
        model = joblib.load(filepath)
        if model_name not in self.results:
            self.results[model_name] = {}
        self.results[model_name]['model'] = model
        
        return model
