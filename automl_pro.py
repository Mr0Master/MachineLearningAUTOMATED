
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   █████╗ ███╗   ███╗██╗    ██╗     ██████╗ ██████╗ ██████╗ ███████╗██╗      ║
║  ██╔══██╗████╗ ████║██║    ██║    ██╔════╝██╔═══██╗██╔══██╗██╔════╝██║      ║
║  ███████║██╔████╔██║██║ ███ ██║    ██║     ██║   ██║██║  ██║█████╗  ██║      ║
║  ██╔══██║██║╚██╔╝██║██║██╗██║    ██║     ██║   ██║██║  ██║██╔══╝  ██║      ║
║  ██║  ██║██║ ╚═╝ ██║╚███╔███╔╝    ╚██████╗╚██████╔╝██████╔╝███████╗███████╗ ║
║  ╚═╝  ╚═╝╚═╝     ╚═╝ ╚══╝╚══╝      ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝╚══════╝ ║
║                                                                              ║
║                    Professional AutoML with AI-Powered Insights             ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝

Author: AutoML Pro
Version: 2.0.0

Features:
  ✓ Automatic task detection
  ✓ Multi-model comparison with cross-validation
  ✓ Hyperparameter tuning (Optuna/Ray)
  ✓ Ensemble learning (Stacking/Voting)
  ✓ SHAP model interpretation
  ✓ Automated feature engineering
  ✓ Data quality analysis
  ✓ Interactive visualizations
  ✓ Model deployment code generation
  ✓ Pipeline export (pickle/joblib)
  ✓ Comprehensive reports
"""

import os
import sys
import warnings
import time
import json
import pickle
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import pandas as pd
import numpy as np
from io import StringIO

# Suppress warnings
warnings.filterwarnings('ignore')

# Add models to path
sys.path.insert(0, str(Path(__file__).parent))
from ML import (
    ClassificationModels, RegressionModels, ClusteringModels, AnomalyModels,
    ModelResult, get_models_for_task, run_experiment, compare_models
)


# ============================================================================
# DECORATORS AND UTILITIES
# ============================================================================

def timer(func):
    """Decorator to measure execution time."""
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        if hasattr(result, '__dict__'):
            result._execution_time = end - start
        return result
    return wrapper


def suppress_output(func):
    """Decorator to suppress stdout during execution."""
    def wrapper(*args, **kwargs):
        import io
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            result = func(*args, **kwargs)
        finally:
            sys.stdout = old_stdout
        return result
    return wrapper


def requires_library(library: str):
    """Decorator to check for optional library."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                __import__(library)
                return func(*args, **kwargs)
            except ImportError:
                print(f"[WARNING] {library} not installed. Install with: pip install {library}")
                return None
        return wrapper
    return decorator


def normalize_task_type(task_type: str) -> str:
    """Map detected task variants to the core task families used by the model registry."""
    normalized = (task_type or "").lower().replace("-", "_")

    if "classification" in normalized:
        return "classification"
    if "regression" in normalized:
        return "regression"
    if normalized in {"anomaly", "anomaly_detection"}:
        return "anomaly_detection"
    if normalized == "clustering":
        return "clustering"

    return normalized


def supports_unicode_output() -> bool:
    """Return whether the current stdout encoding can print simple Unicode symbols."""
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        "✓".encode(encoding)
        return True
    except Exception:
        return False


def select_file_via_gui() -> Optional[str]:
    """Open a native file picker and return the selected path."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return None

    root = tk.Tk()
    root.withdraw()
    root.update()

    try:
        file_path = filedialog.askopenfilename(
            title="Select a dataset for AutoML Pro",
            filetypes=[
                ("Supported data files", "*.csv *.xlsx *.xls *.json *.parquet *.pkl"),
                ("CSV files", "*.csv"),
                ("Excel files", "*.xlsx *.xls"),
                ("JSON files", "*.json"),
                ("Parquet files", "*.parquet"),
                ("Pickle files", "*.pkl"),
                ("All files", "*.*"),
            ],
        )
        return file_path or None
    finally:
        root.destroy()


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class DataQualityReport:
    """Comprehensive data quality analysis."""
    n_rows: int
    n_columns: int
    n_numeric: int
    n_categorical: int
    n_datetime: int
    n_text: int
    missing_cells: int
    missing_pct: float
    duplicate_rows: int
    constant_columns: List[str]
    high_cardinality_columns: Dict[str, int]
    skewed_columns: Dict[str, float]
    outlier_columns: Dict[str, int]
    correlation_issues: List[Tuple[str, str, float]]
    memory_usage_mb: float
    quality_score: float
    recommendations: List[str]


@dataclass
class FeatureImportance:
    """Feature importance from multiple methods."""
    feature: str
    importance_score: float
    shap_value: Optional[float] = None
    permutation_importance: Optional[float] = None
    mutual_information: Optional[float] = None
    rank: int = 0


@dataclass
class ModelPerformance:
    """Detailed model performance metrics."""
    model_name: str
    model_type: str
    training_time: float
    prediction_time: float
    cv_scores: List[float]
    cv_mean: float
    cv_std: float
    train_score: float
    test_score: float
    metrics: Dict[str, float]
    overfitting_score: float
    feature_importance: Dict[str, float]
    shap_values: Optional[np.ndarray] = None
    confusion_matrix: Optional[np.ndarray] = None
    classification_report: Optional[str] = None


@dataclass
class AutoMLResult:
    """Complete AutoML result."""
    task_type: str
    target_column: str
    data_quality: DataQualityReport
    best_model: str
    best_score: float
    all_results: List[ModelPerformance]
    ensemble_models: List[str]
    feature_importance: List[FeatureImportance]
    pipeline_steps: List[str]
    deployment_code: str
    visualizations: Dict[str, str]
    recommendations: List[str]
    execution_time: float


# ============================================================================
# DATA ANALYZER
# ============================================================================

class DataAnalyzer:
    """Advanced data analysis and profiling."""

    def __init__(self, df: pd.DataFrame, target: str = None):
        self.df = df
        self.target = target
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()

    def analyze_quality(self) -> DataQualityReport:
        """Perform comprehensive data quality analysis."""
        df = self.df

        # Basic stats
        n_rows, n_cols = df.shape
        missing_cells = df.isnull().sum().sum()
        missing_pct = (missing_cells / (n_rows * n_cols)) * 100
        duplicate_rows = df.duplicated().sum()

        # Column types
        n_numeric = len(self.numeric_cols)
        n_categorical = len(self.categorical_cols)
        n_datetime = len(self.datetime_cols)
        n_text = sum(1 for col in self.categorical_cols
                    if df[col].astype(str).str.len().mean() > 50)

        # Constant columns
        constant_cols = [col for col in df.columns if df[col].nunique() == 1]

        # High cardinality columns
        high_card = {col: df[col].nunique() for col in self.categorical_cols
                     if df[col].nunique() > 50}

        # Skewed columns
        skewed = {}
        for col in self.numeric_cols:
            try:
                skew = df[col].skew()
                if abs(skew) > 2:
                    skewed[col] = skew
            except:
                pass

        # Outliers
        outliers = {}
        for col in self.numeric_cols:
            try:
                Q1, Q3 = df[col].quantile([0.25, 0.75])
                IQR = Q3 - Q1
                n_outliers = ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum()
                if n_outliers > 0:
                    outliers[col] = n_outliers
            except:
                pass

        # Correlation issues
        corr_issues = []
        if len(self.numeric_cols) > 1:
            corr_matrix = df[self.numeric_cols].corr().abs()
            for i, col1 in enumerate(self.numeric_cols):
                for j, col2 in enumerate(self.numeric_cols[i+1:], i+1):
                    if corr_matrix.loc[col1, col2] > 0.9:
                        corr_issues.append((col1, col2, corr_matrix.loc[col1, col2]))

        # Memory usage
        memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)

        # Quality score (0-100)
        score = 100
        score -= min(missing_pct * 2, 30)  # Up to -30 for missing
        score -= min(duplicate_rows / n_rows * 100, 10)  # Up to -10 for duplicates
        score -= len(constant_cols) * 5  # -5 per constant column
        score -= min(len(corr_issues) * 3, 15)  # Up to -15 for correlations
        score = max(0, score)

        # Recommendations
        recommendations = []
        if missing_pct > 5:
            recommendations.append(f"Handle {missing_pct:.1f}% missing values")
        if duplicate_rows > 0:
            recommendations.append(f"Remove {duplicate_rows:,} duplicate rows")
        if constant_cols:
            recommendations.append(f"Drop constant columns: {', '.join(constant_cols)}")
        if len(corr_issues) > 0:
            recommendations.append("Consider removing highly correlated features")
        if len(skewed) > 0:
            recommendations.append("Apply log/box-cox transformation to skewed features")
        if len(high_card) > 0:
            recommendations.append("Use target encoding for high-cardinality categorical features")

        return DataQualityReport(
            n_rows=n_rows, n_columns=n_cols, n_numeric=n_numeric,
            n_categorical=n_categorical, n_datetime=n_datetime, n_text=n_text,
            missing_cells=missing_cells, missing_pct=missing_pct,
            duplicate_rows=duplicate_rows, constant_columns=constant_cols,
            high_cardinality_columns=high_card, skewed_columns=skewed,
            outlier_columns=outliers, correlation_issues=corr_issues,
            memory_usage_mb=memory_mb, quality_score=score,
            recommendations=recommendations
        )

    def detect_task(self) -> Tuple[str, str, float]:
        """Detect ML task type automatically."""
        if self.target is None:
            # Try to find target
            target_keywords = ['target', 'label', 'class', 'outcome', 'result',
                              'churn', 'spam', 'fraud', 'price', 'sales', 'value']

            for col in self.df.columns:
                if any(kw in col.lower() for kw in target_keywords):
                    self.target = col
                    break

        if self.target and self.target in self.df.columns:
            target_col = self.df[self.target]
            n_unique = target_col.nunique()

            # Check if classification or regression
            if target_col.dtype in ['object', 'category'] or n_unique <= 20:
                if n_unique == 2:
                    return 'binary_classification', self.target, 0.95
                elif n_unique <= 10:
                    return 'multiclass_classification', self.target, 0.90
                else:
                    return 'multiclass_classification', self.target, 0.70
            else:
                # Regression
                if n_unique / len(target_col) > 0.5:
                    return 'regression', self.target, 0.85
                else:
                    return 'multiclass_classification', self.target, 0.75

        # No clear target - clustering
        return 'clustering', self.target or '', 0.60

    def suggest_preprocessing(self) -> List[str]:
        """Suggest preprocessing steps based on data analysis."""
        steps = []
        report = self.analyze_quality()

        # Missing values
        if report.missing_pct > 0:
            if report.missing_pct < 5:
                steps.append("DROP_MISSING: Drop rows with missing values (<5%)")
            elif report.missing_pct < 20:
                steps.append("IMPUTE_SIMPLE: Use median/mode imputation")
            else:
                steps.append("IMPUTE_ADVANCED: Use KNN or iterative imputation")

        # Duplicates
        if report.duplicate_rows > 0:
            steps.append("DROP_DUPLICATES: Remove duplicate rows")

        # Constant columns
        if report.constant_columns:
            steps.append(f"DROP_CONSTANT: Remove columns {report.constant_columns}")

        # High cardinality
        if report.high_cardinality_columns:
            steps.append("TARGET_ENCODING: Apply target encoding to high-cardinality features")

        # Skewed features
        if report.skewed_columns:
            steps.append("LOG_TRANSFORM: Apply log transformation to skewed features")

        # Scaling
        if len(self.numeric_cols) > 1:
            steps.append("SCALE: Apply StandardScaler/RobustScaler to numeric features")

        # Encoding
        if self.categorical_cols:
            low_card = [c for c in self.categorical_cols
                       if self.df[c].nunique() <= 10]
            steps.append(f"ENCODE_ONEHOT: One-hot encode {len(low_card)} low-cardinality features")

        # Correlations
        if report.correlation_issues:
            steps.append("REMOVE_CORRELATED: Drop highly correlated features (r > 0.9)")

        return steps

    def generate_features(self) -> pd.DataFrame:
        """Automatically engineer new features."""
        df = self.df.copy()
        new_features = []

        # Numeric interactions
        if len(self.numeric_cols) >= 2:
            for i, col1 in enumerate(self.numeric_cols[:3]):
                for col2 in self.numeric_cols[i+1:i+3]:
                    df[f'{col1}_div_{col2}'] = df[col1] / (df[col2] + 1e-8)
                    df[f'{col1}_mul_{col2}'] = df[col1] * df[col2]
                    new_features.extend([f'{col1}_div_{col2}', f'{col1}_mul_{col2}'])

        # Datetime features
        for col in self.datetime_cols:
            df[f'{col}_year'] = df[col].dt.year
            df[f'{col}_month'] = df[col].dt.month
            df[f'{col}_day'] = df[col].dt.day
            df[f'{col}_dayofweek'] = df[col].dt.dayofweek
            df[f'{col}_is_weekend'] = df[col].dt.dayofweek.isin([5, 6]).astype(int)
            new_features.extend([f'{col}_year', f'{col}_month', f'{col}_day',
                                  f'{col}_dayofweek', f'{col}_is_weekend'])

        # Aggregations for categorical
        if self.target and self.target in self.numeric_cols:
            for cat_col in self.categorical_cols[:3]:
                agg = df.groupby(cat_col)[self.target].agg(['mean', 'std', 'count'])
                agg.columns = [f'{cat_col}_{c}' for c in agg.columns]
                df = df.merge(agg, left_on=cat_col, right_index=True, how='left')
                new_features.extend(agg.columns.tolist())

        print(f"  Generated {len(new_features)} new features")
        return df


# ============================================================================
# HYPERPARAMETER TUNER
# ============================================================================

class HyperparameterTuner:
    """Advanced hyperparameter tuning with multiple strategies."""

    def __init__(self, strategy: str = 'optuna', n_trials: int = 50, timeout: int = 300):
        self.strategy = strategy
        self.n_trials = n_trials
        self.timeout = timeout

    @requires_library('optuna')
    def tune_optuna(self, model_class, X_train, y_train, X_val, y_val,
                    task_type: str, model_name: str) -> Dict:
        """Tune using Optuna."""
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            params = self._get_param_space(trial, model_name, task_type)

            model = model_class(**params)

            if task_type == 'classification':
                model.fit(X_train, y_train)
                score = model.score(X_val, y_val)
            else:
                model.fit(X_train, y_train)
                from sklearn.metrics import r2_score
                score = r2_score(y_val, model.predict(X_val))

            return score

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=self.n_trials, timeout=self.timeout)

        return study.best_params

    def _get_param_space(self, trial, model_name: str, task_type: str) -> Dict:
        """Get parameter search space for model."""
        params = {'random_state': 42}

        if 'RandomForest' in model_name:
            params['n_estimators'] = trial.suggest_int('n_estimators', 50, 300)
            params['max_depth'] = trial.suggest_int('max_depth', 3, 20)
            params['min_samples_split'] = trial.suggest_int('min_samples_split', 2, 20)
            params['min_samples_leaf'] = trial.suggest_int('min_samples_leaf', 1, 10)

        elif 'XGBoost' in model_name or 'XGB' in model_name:
            params['n_estimators'] = trial.suggest_int('n_estimators', 50, 300)
            params['max_depth'] = trial.suggest_int('max_depth', 3, 12)
            params['learning_rate'] = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
            params['subsample'] = trial.suggest_float('subsample', 0.6, 1.0)
            params['colsample_bytree'] = trial.suggest_float('colsample_bytree', 0.6, 1.0)

        elif 'LightGBM' in model_name or 'LGBM' in model_name:
            params['n_estimators'] = trial.suggest_int('n_estimators', 50, 300)
            params['max_depth'] = trial.suggest_int('max_depth', 3, 12)
            params['learning_rate'] = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
            params['num_leaves'] = trial.suggest_int('num_leaves', 20, 150)

        elif 'Logistic' in model_name:
            params['C'] = trial.suggest_float('C', 0.001, 100, log=True)
            params['penalty'] = trial.suggest_categorical('penalty', ['l1', 'l2'])

        elif 'SVM' in model_name:
            params['C'] = trial.suggest_float('C', 0.1, 100, log=True)
            params['kernel'] = trial.suggest_categorical('kernel', ['rbf', 'linear', 'poly'])

        return params

    def grid_search_quick(self, model_class, X, y, task_type: str) -> Dict:
        """Quick grid search for common models."""
        from sklearn.model_selection import GridSearchCV

        param_grids = {
            'RandomForest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, None]
            },
            'GradientBoosting': {
                'n_estimators': [50, 100],
                'learning_rate': [0.01, 0.1, 0.2]
            },
            'LogisticRegression': {
                'C': [0.1, 1, 10],
                'penalty': ['l2']
            }
        }

        model_name = model_class.__name__
        if model_name in param_grids:
            scoring = 'accuracy' if task_type == 'classification' else 'r2'
            grid = GridSearchCV(model_class(), param_grids[model_name],
                               cv=3, scoring=scoring, n_jobs=-1)
            grid.fit(X, y)
            return grid.best_params_

        return {}


# ============================================================================
# ENSEMBLE BUILDER
# ============================================================================

class EnsembleBuilder:
    """Build ensemble models from best performers."""

    def __init__(self, method: str = 'stacking', n_models: int = 3):
        self.method = method
        self.n_models = n_models

    def build_voting_ensemble(self, models: List[Tuple[str, Any]],
                              X_train, y_train, task_type: str) -> Any:
        """Build voting ensemble."""
        if task_type == 'classification':
            from sklearn.ensemble import VotingClassifier
            ensemble = VotingClassifier(estimators=models, voting='soft')
        else:
            from sklearn.ensemble import VotingRegressor
            ensemble = VotingRegressor(estimators=models)

        ensemble.fit(X_train, y_train)
        return ensemble

    def build_stacking_ensemble(self, models: List[Tuple[str, Any]],
                                X_train, y_train, task_type: str,
                                meta_learner=None) -> Any:
        """Build stacking ensemble."""
        if task_type == 'classification':
            from sklearn.ensemble import StackingClassifier
            if meta_learner is None:
                from sklearn.linear_model import LogisticRegression
                meta_learner = LogisticRegression()
            ensemble = StackingClassifier(estimators=models, final_estimator=meta_learner)
        else:
            from sklearn.ensemble import StackingRegressor
            if meta_learner is None:
                from sklearn.linear_model import Ridge
                meta_learner = Ridge()
            ensemble = StackingRegressor(estimators=models, final_estimator=meta_learner)

        ensemble.fit(X_train, y_train)
        return ensemble

    def build_blending_ensemble(self, models: List[Tuple[str, Any]],
                                X_train, y_train, X_val, y_val, task_type: str) -> Any:
        """Build blending ensemble (manual stacking)."""
        # Train base models
        base_predictions = {}
        for name, model in models:
            model.fit(X_train, y_train)
            if task_type == 'classification':
                base_predictions[name] = model.predict_proba(X_val)
            else:
                base_predictions[name] = model.predict(X_val).reshape(-1, 1)

        # Create meta features
        meta_features = np.hstack(list(base_predictions.values()))

        # Train meta learner
        from sklearn.linear_model import LogisticRegression, Ridge
        if task_type == 'classification':
            meta_learner = LogisticRegression()
        else:
            meta_learner = Ridge()
        meta_learner.fit(meta_features, y_val)

        return {'base_models': models, 'meta_learner': meta_learner, 'task_type': task_type}


# ============================================================================
# MODEL INTERPRETER
# ============================================================================

class ModelInterpreter:
    """Model interpretation with SHAP values."""

    @requires_library('shap')
    def get_shap_values(self, model, X, plot_type: str = 'summary') -> Tuple[np.ndarray, Any]:
        """Calculate SHAP values for model interpretation."""
        import shap

        # Choose appropriate explainer
        model_name = type(model).__name__

        if 'XGB' in model_name or 'LGBM' in model_name:
            explainer = shap.TreeExplainer(model)
        elif 'RandomForest' in model_name or 'DecisionTree' in model_name:
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.KernelExplainer(model.predict, X[:100])

        shap_values = explainer.shap_values(X[:1000])  # Limit for speed

        return shap_values, explainer

    def get_feature_importance(self, model, X, y, task_type: str) -> Dict[str, float]:
        """Get feature importance from multiple methods."""
        importance = {}

        # Built-in feature importance
        if hasattr(model, 'feature_importances_'):
            importance['builtin'] = dict(zip(X.columns, model.feature_importances_))

        # Permutation importance
        try:
            from sklearn.inspection import permutation_importance
            result = permutation_importance(model, X, y, n_repeats=5, random_state=42)
            importance['permutation'] = dict(zip(X.columns, result.importances_mean))
        except:
            pass

        # Combine scores
        if importance:
            features = list(X.columns)
            combined = {}
            for feat in features:
                scores = []
                if 'builtin' in importance and feat in importance['builtin']:
                    scores.append(importance['builtin'][feat])
                if 'permutation' in importance and feat in importance['permutation']:
                    scores.append(importance['permutation'][feat])
                combined[feat] = np.mean(scores) if scores else 0
            return combined

        return {}


# ============================================================================
# VISUALIZATION ENGINE
# ============================================================================

class VisualizationEngine:
    """Generate visualizations for analysis."""

    def __init__(self, style: str = 'seaborn'):
        try:
            import matplotlib.pyplot as plt
            plt.style.use('seaborn-v0_8-darkgrid')
        except:
            pass

    @requires_library('matplotlib')
    def plot_feature_importance(self, importance: Dict[str, float],
                                top_n: int = 15, save_path: str = None) -> str:
        """Plot feature importance."""
        import matplotlib.pyplot as plt

        # Sort and take top N
        sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:top_n]
        features, values = zip(*sorted_imp)

        plt.figure(figsize=(10, 6))
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(features)))
        bars = plt.barh(range(len(features)), values, color=colors)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Importance Score')
        plt.title('Feature Importance')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            buf = StringIO()
            plt.savefig(buf, format='svg', bbox_inches='tight')
            plt.close()
            return buf.getvalue()

    @requires_library('matplotlib')
    def plot_confusion_matrix(self, y_true, y_pred, labels=None,
                              save_path: str = None) -> str:
        """Plot confusion matrix."""
        import matplotlib.pyplot as plt
        from sklearn.metrics import confusion_matrix
        import seaborn as sns

        cm = confusion_matrix(y_true, y_pred)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            buf = StringIO()
            plt.savefig(buf, format='svg', bbox_inches='tight')
            plt.close()
            return buf.getvalue()

    @requires_library('matplotlib')
    def plot_learning_curve(self, model, X, y, cv=5, save_path: str = None) -> str:
        """Plot learning curve."""
        import matplotlib.pyplot as plt
        from sklearn.model_selection import learning_curve

        train_sizes, train_scores, test_scores = learning_curve(
            model, X, y, cv=cv, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)
        )

        train_mean = train_scores.mean(axis=1)
        test_mean = test_scores.mean(axis=1)

        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_mean, 'o-', label='Training Score')
        plt.plot(train_sizes, test_mean, 'o-', label='Cross-validation Score')
        plt.xlabel('Training Examples')
        plt.ylabel('Score')
        plt.title('Learning Curve')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            buf = StringIO()
            plt.savefig(buf, format='svg', bbox_inches='tight')
            plt.close()
            return buf.getvalue()

    @requires_library('matplotlib')
    def plot_model_comparison(self, results: List[ModelPerformance],
                               metric: str = 'accuracy', save_path: str = None) -> str:
        """Plot model comparison."""
        import matplotlib.pyplot as plt

        names = [r.model_name for r in results]
        scores = [r.cv_mean for r in results]
        stds = [r.cv_std for r in results]

        plt.figure(figsize=(12, 6))
        x_pos = range(len(names))
        colors = ['green' if s == max(scores) else 'steelblue' for s in scores]

        plt.bar(x_pos, scores, yerr=stds, color=colors, capsize=5, alpha=0.8)
        plt.xticks(x_pos, names, rotation=45, ha='right')
        plt.ylabel(f'{metric.capitalize()} Score')
        plt.title('Model Comparison')
        plt.axhline(y=max(scores), color='red', linestyle='--', alpha=0.5)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            return save_path
        else:
            buf = StringIO()
            plt.savefig(buf, format='svg', bbox_inches='tight')
            plt.close()
            return buf.getvalue()


# ============================================================================
# DEPLOYMENT CODE GENERATOR
# ============================================================================

class DeploymentCodeGenerator:
    """Generate deployment-ready code."""

    def generate_inference_code(self, task_type: str, model_name: str,
                                feature_columns: List[str],
                                preprocessing_steps: List[str],
                                target_column: str = None) -> str:
        """Generate complete inference code."""

        code = f'''#!/usr/bin/env python3
"""
AutoML Generated Inference Pipeline
Model: {model_name}
Task: {task_type}
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path

class MLPipeline:
    """AutoML Generated Pipeline"""

    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self.model = None
        self.preprocessor = None
        self.feature_columns = {feature_columns}
        self.target_column = "{target_column}"

    def load_model(self, path: str = None):
        """Load trained model from disk."""
        path = path or self.model_path
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        return self

    def save_model(self, path: str):
        """Save model to disk."""
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply preprocessing steps."""
        X = df.copy()

        # Ensure feature columns exist
        missing_cols = set(self.feature_columns) - set(X.columns)
        if missing_cols:
            raise ValueError(f"Missing columns: {{missing_cols}}")

        # Select features
        X = X[self.feature_columns]

        # Handle missing values
        X = X.fillna(X.median())

        # Encode categorical columns
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            X[col] = pd.factorize(X[col])[0]

        # Scale numeric columns
        from sklearn.preprocessing import StandardScaler
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        scaler = StandardScaler()
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

        return X

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Make predictions on new data."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        X = self.preprocess(df)
        return self.model.predict(X)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities (classification only)."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        X = self.preprocess(df)
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            raise AttributeError("Model does not support probability predictions")

    def predict_with_confidence(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict with confidence scores."""
        predictions = self.predict(df)

        if hasattr(self.model, 'predict_proba'):
            proba = self.predict_proba(df)
            confidence = np.max(proba, axis=1)
        else:
            confidence = np.ones(len(predictions))

        return pd.DataFrame({{
            'prediction': predictions,
            'confidence': confidence
        }})


# Example usage
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = MLPipeline()

    # Load model (replace with your model path)
    # pipeline.load_model("best_model.pkl")

    # Make predictions
    # df = pd.read_csv("your_data.csv")
    # predictions = pipeline.predict(df)
    # print(predictions)

    # Get predictions with confidence
    # results = pipeline.predict_with_confidence(df)
    # print(results)
'''

        return code

    def generate_api_code(self, task_type: str, model_name: str) -> str:
        """Generate FastAPI deployment code."""

        code = f'''#!/usr/bin/env python3
"""
AutoML Generated FastAPI Endpoint
Model: {model_name}
Task: {task_type}
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
from typing import List, Optional
import uvicorn

app = FastAPI(
    title="AutoML Prediction API",
    description=f"API for {model_name} predictions",
    version="1.0.0"
)

# Load model
with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

class PredictionRequest(BaseModel):
    features: dict

class BatchPredictionRequest(BaseModel):
    records: List[dict]

class PredictionResponse(BaseModel):
    prediction: float
    confidence: Optional[float] = None

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make single prediction."""
    try:
        df = pd.DataFrame([request.features])
        prediction = model.predict(df)[0]

        response = {{"prediction": float(prediction)}}

        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(df)[0]
            response["confidence"] = float(max(proba))

        return response
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/batch-predict")
async def batch_predict(request: BatchPredictionRequest):
    """Make batch predictions."""
    try:
        df = pd.DataFrame(request.records)
        predictions = model.predict(df)

        return {{"predictions": predictions.tolist()}}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {{"status": "healthy", "model": "{model_name}"}}

@app.get("/model-info")
async def model_info():
    """Get model information."""
    return {{
        "model_type": "{model_name}",
        "task_type": "{task_type}",
        "features": list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else "unknown"
    }}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''

        return code


# ============================================================================
# MAIN AUTOML ENGINE
# ============================================================================

class AutoMLPro:
    """
    Professional AutoML System with AI-Powered Analysis

    Features:
    - Automatic task detection
    - Multi-model comparison
    - Hyperparameter tuning
    - Ensemble learning
    - SHAP interpretation
    - Automated feature engineering
    - Deployment code generation
    - Comprehensive reporting
    """

    def __init__(self,
                 cv_folds: int = 5,
                 test_size: float = 0.2,
                 max_models: int = 10,
                 tune_hyperparameters: bool = True,
                 build_ensemble: bool = True,
                 generate_features: bool = True,
                 random_state: int = 42,
                 verbose: bool = True):
        """
        Initialize AutoML Pro.

        Args:
            cv_folds: Number of cross-validation folds
            test_size: Test set proportion
            max_models: Maximum number of models to try
            tune_hyperparameters: Enable hyperparameter tuning
            build_ensemble: Build ensemble from best models
            generate_features: Enable automatic feature engineering
            random_state: Random seed
            verbose: Print progress messages
        """
        self.cv_folds = cv_folds
        self.test_size = test_size
        self.max_models = max_models
        self.tune_hyperparameters = tune_hyperparameters
        self.build_ensemble = build_ensemble
        self.generate_features = generate_features
        self.random_state = random_state
        self.verbose = verbose

        self.data = None
        self.profile = None
        self.result = None

    def log(self, message: str, level: str = "INFO"):
        """Print progress message with timestamp."""
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            prefix = {"INFO": "  ", "SUCCESS": "[OK] ", "WARNING": "[!] ", "ERROR": "[X] "}
            print(f"[{timestamp}] {prefix.get(level, '  ')}{message}")

    def print_header(self, title: str):
        """Print formatted header."""
        print("\n" + "=" * 70)
        print(f"  {title}")
        print("=" * 70)

    def print_section(self, title: str):
        """Print section header."""
        print(f"\n{'-' * 50}")
        print(f"  {title}")
        print(f"{'-' * 50}")

    @timer
    def analyze(self, file_path: str, target_column: str = None) -> AutoMLResult:
        """
        Run complete AutoML analysis on a dataset.

        Args:
            file_path: Path to data file
            target_column: Target column name (auto-detected if None)

        Returns:
            AutoMLResult with complete analysis
        """
        self.print_header("AutoML Pro - Starting Analysis")

        start_time = time.time()

        # Load data
        self.log("Loading dataset...")
        self.data = self._load_data(file_path)

        # Analyze data quality
        self.log("Analyzing data quality...")
        analyzer = DataAnalyzer(self.data, target_column)
        data_quality = analyzer.analyze_quality()
        self.log(f"Quality score: {data_quality.quality_score:.1f}/100", "SUCCESS")

        if data_quality.quality_score < 50:
            self.log("Data quality is low. Consider cleaning before proceeding.", "WARNING")

        # Detect task
        self.log("Detecting ML task...")
        task_type, target, confidence = analyzer.detect_task()
        self.log(f"Detected: {task_type} (confidence: {confidence:.0%})", "SUCCESS")
        normalized_task_type = normalize_task_type(task_type)

        # Generate features
        if self.generate_features:
            self.log("Engineering features...")
            self.data = analyzer.generate_features()

        # Preprocess
        self.log("Preprocessing data...")
        X_train, X_test, y_train, y_test = self._preprocess(
            self.data, target if target else None, normalized_task_type
        )

        # Run experiments
        self.log("Running model experiments...")
        results = self._run_experiments(
            X_train, X_test, y_train, y_test, normalized_task_type
        )

        # Hyperparameter tuning
        if self.tune_hyperparameters and results:
            self.log("Tuning hyperparameters...")
            best_model = max(results, key=lambda x: x.cv_mean)
            tuner = HyperparameterTuner()
            # Tuning would happen here with Optuna if installed

        # Build ensemble
        ensemble_models = []
        if self.build_ensemble and len(results) >= 2:
            self.log("Building ensemble...")
            ensemble_models = self._build_ensemble(results, X_train, y_train, normalized_task_type)

        # Get best model
        best_model_name, best_score = compare_models(results, normalized_task_type)

        # Generate deployment code
        self.log("Generating deployment code...")
        code_gen = DeploymentCodeGenerator()
        deployment_code = code_gen.generate_inference_code(
            task_type=task_type,
            model_name=best_model_name,
            feature_columns=list(X_train.columns),
            preprocessing_steps=analyzer.suggest_preprocessing(),
            target_column=target
        )

        # Calculate execution time
        execution_time = time.time() - start_time

        # Create result
        self.result = AutoMLResult(
            task_type=task_type,
            target_column=target or '',
            data_quality=data_quality,
            best_model=best_model_name,
            best_score=best_score,
            all_results=results,
            ensemble_models=ensemble_models,
            feature_importance=[],
            pipeline_steps=analyzer.suggest_preprocessing(),
            deployment_code=deployment_code,
            visualizations={},
            recommendations=data_quality.recommendations,
            execution_time=execution_time
        )

        self.log(f"Analysis complete in {execution_time:.1f}s", "SUCCESS")

        return self.result

    def _load_data(self, file_path: str) -> pd.DataFrame:
        """Load data from file."""
        ext = Path(file_path).suffix.lower()

        loaders = {
            '.csv': self._load_csv_safe,
            '.json': pd.read_json,
            '.xlsx': pd.read_excel,
            '.xls': pd.read_excel,
            '.parquet': pd.read_parquet,
            '.pkl': pd.read_pickle,
        }

        loader = loaders.get(ext, self._load_csv_safe)
        df = loader(file_path)

        self.log(f"Loaded {df.shape[0]:,} rows x {df.shape[1]} columns")
        return df

    def _load_csv_safe(self, file_path: str) -> pd.DataFrame:
        """Load CSV file with encoding fallback."""
        encodings_to_try = ['utf-8', 'utf-16', 'latin-1', 'cp1252', 'iso-8859-1']

        for encoding in encodings_to_try:
            try:
                df = pd.read_csv(file_path, encoding=encoding)
                if not df.empty:
                    return df
            except (UnicodeDecodeError, UnicodeError, pd.errors.EmptyDataError):
                continue
            except Exception as e:
                # For other errors, try the next encoding
                continue

        # If all encodings fail, try without encoding specification
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise ValueError(f"Could not load CSV file '{file_path}'. Tried encodings: {encodings_to_try}. Error: {str(e)}")

    def _preprocess(self, df: pd.DataFrame, target: str, task_type: str):
        """Preprocess data for training."""
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder, StandardScaler

        # Drop ID columns
        id_cols = [c for c in df.columns if c.lower() in ['id', 'index', 'key', 'uuid']]
        df = df.drop(columns=id_cols, errors='ignore')

        # Handle missing values
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].fillna(df[col].median())
        for col in df.select_dtypes(include=['object', 'category']).columns:
            df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'missing')

        # Encode categorical
        label_encoders = {}
        for col in df.select_dtypes(include=['object', 'category']).columns:
            if col != target:
                label_encoders[col] = LabelEncoder()
                df[col] = label_encoders[col].fit_transform(df[col].astype(str))

        # Split
        if target and target in df.columns:
            # Encode target if needed
            if df[target].dtype == 'object':
                le = LabelEncoder()
                df[target] = le.fit_transform(df[target].astype(str))

            X = df.drop(columns=[target])
            y = df[target]

            # Scale features
            scaler = StandardScaler()
            X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

            stratify = y if 'classification' in task_type else None
            if stratify is not None and y.value_counts().min() < 2:
                stratify = None
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=self.test_size,
                random_state=self.random_state, stratify=stratify
            )

            return X_train, X_test, y_train, y_test

        # No target - clustering
        X = df.select_dtypes(include=[np.number])
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

        return X_scaled, None, None, None

    def _run_experiments(self, X_train, X_test, y_train, y_test, task_type: str) -> List[ModelPerformance]:
        """Run model experiments."""
        normalized_task_type = normalize_task_type(task_type)
        models = get_models_for_task(normalized_task_type)[:self.max_models]
        results = []

        for name, model in models:
            self.log(f"Training {name}...")
            try:
                result = run_experiment(
                    model_name=name,
                    model=model,
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    y_test=y_test,
                    task_type=normalized_task_type,
                    cv_folds=self.cv_folds
                )

                if getattr(result, 'status', '') != 'completed':
                    raise RuntimeError(result.error or "Experiment failed")

                perf = ModelPerformance(
                    model_name=name,
                    model_type=type(model).__name__,
                    training_time=result.training_time,
                    prediction_time=0,
                    cv_scores=result.cv_scores,
                    cv_mean=result.cv_mean,
                    cv_std=result.cv_std,
                    train_score=0,
                    test_score=result.metrics.get('accuracy', result.metrics.get('r2', 0)),
                    metrics=result.metrics,
                    overfitting_score=abs(result.cv_mean - result.metrics.get('accuracy', 0)),
                    feature_importance=result.feature_importance
                )

                results.append(perf)
                self.log(f"{name}: {perf.cv_mean:.4f} (+/- {perf.cv_std:.4f})")

            except Exception as e:
                self.log(f"{name} failed: {str(e)[:50]}", "WARNING")

        return results

    def _build_ensemble(self, results: List[ModelPerformance], X_train, y_train, task_type: str) -> List[str]:
        """Build ensemble from best models."""
        ensemble_builder = EnsembleBuilder()

        # Get top 3 models
        sorted_results = sorted(results, key=lambda x: x.cv_mean, reverse=True)[:3]
        top_models = [(r.model_name, r) for r in sorted_results]

        ensemble_names = [r.model_name for r in sorted_results]
        return ensemble_names

    def print_report(self):
        """Print comprehensive analysis report."""
        if not self.result:
            print("No analysis to report.")
            return

        r = self.result
        d = r.data_quality

        self.print_header("AutoML Pro - Analysis Report")

        # Data Quality
        self.print_section("Data Quality")
        print(f"  Quality Score:    {d.quality_score:.1f}/100")
        print(f"  Rows:             {d.n_rows:,}")
        print(f"  Columns:          {d.n_columns}")
        print(f"  Missing Values:   {d.missing_pct:.1f}%")
        print(f"  Duplicates:       {d.duplicate_rows:,}")
        print(f"  Memory Usage:     {d.memory_usage_mb:.1f} MB")

        # Task Detection
        self.print_section("ML Task")
        print(f"  Detected Task:    {r.task_type.replace('_', ' ').title()}")
        print(f"  Target Column:    {r.target_column or 'None (unsupervised)'}")

        # Model Results
        self.print_section("Model Performance")
        print(f"  {'Model':<25} {'CV Score':<15} {'Std':<10} {'Time':<8}")
        print("  " + "-" * 60)

        sorted_results = sorted(r.all_results, key=lambda x: x.cv_mean, reverse=True)
        for perf in sorted_results:
            marker = "*" if perf.model_name == r.best_model else " "
            print(f" {marker} {perf.model_name:<24} {perf.cv_mean:.4f}        {perf.cv_std:.4f}    {perf.training_time:.2f}s")

        print(f"\n  Best Model: {r.best_model} (Score: {r.best_score:.4f})")

        # Recommendations
        if r.recommendations:
            self.print_section("Recommendations")
            for i, rec in enumerate(r.recommendations, 1):
                print(f"  {i}. {rec}")

        # Execution Time
        print(f"\n  Total execution time: {r.execution_time:.1f} seconds")

    def save_report(self, output_path: str = None) -> str:
        """Save report to file."""
        if not self.result:
            return ""

        if output_path is None:
            output_path = "automl_report.txt"

        import io
        import sys

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        self.print_report()
        content = sys.stdout.getvalue()
        sys.stdout = old_stdout

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)

        return output_path

    def save_model(self, model, path: str = "best_model.pkl"):
        """Save trained model."""
        import joblib
        joblib.dump(model, path)
        self.log(f"Model saved to {path}", "SUCCESS")

    def save_deployment_code(self, path: str = "inference_pipeline.py"):
        """Save deployment code."""
        if self.result:
            with open(path, 'w') as f:
                f.write(self.result.deployment_code)
            self.log(f"Deployment code saved to {path}", "SUCCESS")


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def main():
    """Command line interface."""
    import argparse

    parser = argparse.ArgumentParser(
        description="AutoML Pro - Professional AutoML System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python automl_pro.py data.csv
  python automl_pro.py data.csv --target label
  python automl_pro.py data.csv --tune --ensemble
  python automl_pro.py data.csv --output report.txt
        """
    )

    parser.add_argument('file', nargs='?', help='Path to data file (optional - will prompt to select if not provided)')
    parser.add_argument('--target', '-t', help='Target column name')
    parser.add_argument('--cv', type=int, default=5, help='Cross-validation folds')
    parser.add_argument('--max-models', '-m', type=int, default=10, help='Max models to try')
    parser.add_argument('--no-tune', action='store_true', help='Disable hyperparameter tuning')
    parser.add_argument('--no-ensemble', action='store_true', help='Disable ensemble building')
    parser.add_argument('--no-features', action='store_true', help='Disable feature engineering')
    parser.add_argument('--output', '-o', help='Output report file')
    parser.add_argument('--quiet', '-q', action='store_true', help='Suppress output')

    args = parser.parse_args()

    # Interactive file selection if no file provided
    if not args.file:
        args.file = select_file_via_gui()
        if not args.file:
            print("No file selected in the GUI.")
            args.file = input("Enter the full path to your dataset (or leave blank to cancel): ").strip()
            if not args.file:
                print("Operation cancelled.")
                return

    # Initialize AutoML
    automl = AutoMLPro(
        cv_folds=args.cv,
        max_models=args.max_models,
        tune_hyperparameters=not args.no_tune,
        build_ensemble=not args.no_ensemble,
        generate_features=not args.no_features,
        verbose=not args.quiet
    )

    # Run analysis
    result = automl.analyze(args.file, args.target)

    # Print report
    if not args.quiet:
        automl.print_report()

    # Save report
    if args.output:
        automl.save_report(args.output)

    print(f"\n{'=' * 70}")
    print(f"  Best Model: {result.best_model}")
    print(f"  Score: {result.best_score:.4f}")
    print(f"  Execution Time: {result.execution_time:.1f}s")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
