"""
Simple Model Evaluation Script for House Price Prediction

This script answers two key questions from the project requirement:
1. How well will the model generalize to new data?
2. Has the model appropriately fit the dataset? (overfitting/underfitting check)

Based on create_model.py with focus on practical business insights.
"""

import json
import pickle
import warnings
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn import model_selection, metrics

warnings.filterwarnings('ignore')

SALES_PATH = "data/kc_house_data.csv"
DEMOGRAPHICS_PATH = "data/zipcode_demographics.csv"
MODEL_PATH = "model/model.pkl"
FEATURES_PATH = "model/model_features.json"

SALES_COLUMN_SELECTION = [
    'price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
    'sqft_above', 'sqft_basement', 'zipcode'
]

def load_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Load and prepare the training data (same as create_model.py)."""
    print("📊 Loading training data...")
    
    data = pd.read_csv(SALES_PATH,
                       usecols=SALES_COLUMN_SELECTION,
                       dtype={'zipcode': str})
    
    demographics = pd.read_csv(DEMOGRAPHICS_PATH,
                               dtype={'zipcode': str})

    merged_data = data.merge(demographics, how="left", on="zipcode").drop(columns="zipcode")
    
    y = merged_data.pop('price')
    X = merged_data
    
    print(f"   • Training samples: {len(X):,}")
    print(f"   • Features: {len(X.columns)}")
    print(f"   • Price range: ${y.min():,.0f} - ${y.max():,.0f}")
    
    return X, y

def load_trained_model():
    """Load the trained model and features."""
    print("\n🤖 Loading trained model...")
    
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    
    with open(FEATURES_PATH, 'r') as f:
        features = json.load(f)
    
    print(f"   • Model type: {type(model).__name__}")
    print(f"   • Features: {len(features)}")
    
    return model, features

def evaluate_generalization(X, y, model):
    """
    Answer: How well will the model generalize to new data?
    Method: Train/test split to simulate unseen data
    """
    print("\n🎯 QUESTION 1: How well will the model generalize to new data?")
    print("-" * 60)
    
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    
    print(f"   • Training on {len(X_train):,} houses")
    print(f"   • Testing on {len(X_test):,} houses (simulating new data)")
    
    model.fit(X_train, y_train)
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_mae = metrics.mean_absolute_error(y_train, y_train_pred)
    test_mae = metrics.mean_absolute_error(y_test, y_test_pred)
    
    train_r2 = metrics.r2_score(y_train, y_train_pred)
    test_r2 = metrics.r2_score(y_test, y_test_pred)
    
    train_mape = np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100
    test_mape = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100
    
    print(f"\n   📈 Performance Metrics:")
    print(f"   • Training MAE: ${train_mae:,.0f} ({train_mape:.1f}% error)")
    print(f"   • Test MAE: ${test_mae:,.0f} ({test_mape:.1f}% error)")
    print(f"   • Training R²: {train_r2:.3f}")
    print(f"   • Test R²: {test_r2:.3f}")
    
    mae_ratio = test_mae / train_mae
    r2_diff = train_r2 - test_r2
    
    print(f"\n   🔍 Generalization Analysis:")
    print(f"   • Test MAE is {mae_ratio:.2f}x training MAE")
    print(f"   • R² drops by {r2_diff:.3f} on test data")
    
    print(f"\n   📊 Interpretation:")
    if mae_ratio <= 1.15 and r2_diff <= 0.08:
        print("   ✅ EXCELLENT: Model generalizes very well to new data!")
        generalization = "Excellent"
    elif mae_ratio <= 1.35 and r2_diff <= 0.15:
        print("   ✅ GOOD: Model should perform well on new houses")
        generalization = "Good"
    elif mae_ratio <= 1.5 and r2_diff <= 0.25:
        print("   ⚠️  FAIR: Model has some generalization challenges")
        generalization = "Fair"
    else:
        print("   ❌ POOR: Model may not generalize well to new data")
        generalization = "Poor"
    
    return {
        'train_mae': train_mae, 'test_mae': test_mae,
        'train_r2': train_r2, 'test_r2': test_r2,
        'train_mape': train_mape, 'test_mape': test_mape,
        'mae_ratio': mae_ratio, 'r2_diff': r2_diff,
        'generalization': generalization
    }

def evaluate_model_fit(X, y, model):
    """
    Answer: Has the model appropriately fit the dataset?
    Method: Cross-validation + overfitting detection
    """
    print("\n🎯 QUESTION 2: Has the model appropriately fit the dataset?")
    print("-" * 60)
    
    print("   📊 Running 5-fold cross-validation...")
    cv_scores = model_selection.cross_val_score(
        model, X, y, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1
    )
    cv_mae_scores = -cv_scores
    
    cv_r2_scores = model_selection.cross_val_score(
        model, X, y, cv=5, scoring='r2', n_jobs=-1
    )
    
    print(f"   • CV MAE: ${cv_mae_scores.mean():,.0f} ± ${cv_mae_scores.std():,.0f}")
    print(f"   • CV R²: {cv_r2_scores.mean():.3f} ± {cv_r2_scores.std():.3f}")
    
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    model.fit(X_train, y_train)
    
    train_mae = metrics.mean_absolute_error(y_train, model.predict(X_train))
    test_mae = metrics.mean_absolute_error(y_test, model.predict(X_test))
    
    mae_ratio = test_mae / train_mae
    
    print(f"\n   🔍 Overfitting Analysis:")
    print(f"   • Training MAE: ${train_mae:,.0f}")
    print(f"   • Test MAE: ${test_mae:,.0f}")
    print(f"   • Ratio (test/train): {mae_ratio:.2f}")
    
    print(f"\n   📊 Model Fit Assessment:")
    if mae_ratio <= 1.15:
        print("   ✅ EXCELLENT FIT: Model is well-balanced, no overfitting")
        model_fit = "Excellent Fit"
    elif mae_ratio <= 1.35:
        print("   ✅ GOOD FIT: Model generalizes well with minimal overfitting")
        model_fit = "Good Fit"
    elif mae_ratio <= 1.5:
        print("   ⚠️  SLIGHT OVERFITTING: Model memorizes training data a bit")
        model_fit = "Slight Overfitting"
    else:
        print("   ❌ OVERFITTING: Model memorizes training data too much")
        model_fit = "Overfitting"
    
    cv_r2_mean = cv_r2_scores.mean()
    if cv_r2_mean >= 0.8:
        performance = "Excellent"
    elif cv_r2_mean >= 0.7:
        performance = "Good"
    elif cv_r2_mean >= 0.6:
        performance = "Fair"
    else:
        performance = "Poor"
    
    print(f"   • Overall Performance: {performance} (R² = {cv_r2_mean:.3f})")
    
    return {
        'cv_mae_mean': cv_mae_scores.mean(),
        'cv_mae_std': cv_mae_scores.std(),
        'cv_r2_mean': cv_r2_mean,
        'cv_r2_std': cv_r2_scores.std(),
        'mae_ratio': mae_ratio,
        'model_fit': model_fit,
        'performance': performance
    }

def price_range_analysis(X, y, model):
    """Analyze how well the model performs across different price ranges."""
    print("\n🎯 BONUS: Performance across price ranges")
    print("-" * 60)
    
    low_price = y <= y.quantile(0.33)
    mid_price = (y > y.quantile(0.33)) & (y <= y.quantile(0.67))
    high_price = y > y.quantile(0.67)
    
    ranges = [
        ("Low Price", low_price, f"${y[low_price].min():,.0f} - ${y[low_price].max():,.0f}"),
        ("Mid Price", mid_price, f"${y[mid_price].min():,.0f} - ${y[mid_price].max():,.0f}"),
        ("High Price", high_price, f"${y[high_price].min():,.0f} - ${y[high_price].max():,.0f}")
    ]
    
    model.fit(X, y)
    predictions = model.predict(X)
    
    print(f"   📊 Performance by Price Range:")
    for name, mask, price_range in ranges:
        if mask.sum() > 0:
            range_mae = metrics.mean_absolute_error(y[mask], predictions[mask])
            range_r2 = metrics.r2_score(y[mask], predictions[mask])
            range_mape = np.mean(np.abs((y[mask] - predictions[mask]) / y[mask])) * 100
            
            print(f"   • {name:10} ({mask.sum():,} houses): MAE=${range_mae:,.0f}, R²={range_r2:.3f}, Error={range_mape:.1f}%")
            print(f"     Range: {price_range}")

def generate_business_recommendations(gen_results, fit_results):
    """Generate actionable recommendations for Sound Realty."""
    print("\n💼 BUSINESS RECOMMENDATIONS FOR SOUND REALTY")
    print("=" * 60)
    
    recommendations = []
    
    if gen_results['generalization'] in ['Excellent', 'Good']:
        recommendations.append("✅ The model is ready for production use on new houses")
    else:
        recommendations.append("⚠️ Consider collecting more training data before deployment")
    
    if fit_results['model_fit'] in ['Overfitting']:
        recommendations.append("⚠️ The model may be too complex - consider simpler approaches")
    elif fit_results['model_fit'] in ['Good Fit', 'Excellent Fit']:
        recommendations.append("✅ Model shows good balance between bias and variance")
    
    if fit_results['performance'] in ['Good', 'Excellent']:
        recommendations.append("✅ Model performance is suitable for business use")
        recommendations.append(f"📊 Expect typical prediction errors of ~${gen_results['test_mae']:,.0f}")
    else:
        recommendations.append("⚠️ Consider feature engineering or different algorithms for better accuracy")
    
    error_pct = gen_results['test_mape']
    if error_pct < 12:
        recommendations.append(f"✅ {error_pct:.1f}% average error is excellent for real estate")
    elif error_pct < 20:
        recommendations.append(f"✅ {error_pct:.1f}% average error is acceptable for real estate")
    else:
        recommendations.append(f"⚠️ {error_pct:.1f}% average error may be too high for some use cases")
    
    recommendations.extend([
        "📈 Consider updating the model quarterly with new sales data",
        "🎯 Monitor prediction accuracy on actual sales to validate performance",
        "🔍 Be cautious with predictions on unusual properties (very high/low prices)"
    ])
    
    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. {rec}")

def main():
    """Main evaluation function."""
    print("🏠 Sound Realty Model Evaluation")
    print("📋 Answering: Will this model work well for new houses?")
    print("=" * 60)
    
    X, y = load_data()
    model, features = load_trained_model()
    
    gen_results = evaluate_generalization(X, y, model)
    fit_results = evaluate_model_fit(X, y, model)
    
    price_range_analysis(X, y, model)
    
    generate_business_recommendations(gen_results, fit_results)
    
    print(f"\n" + "=" * 60)
    print("📋 EXECUTIVE SUMMARY")
    print("=" * 60)
    print(f"✅ Model Generalization: {gen_results['generalization']}")
    print(f"✅ Model Fit: {fit_results['model_fit']}")
    print(f"✅ Overall Performance: {fit_results['performance']}")
    print(f"💰 Expected Error: ${gen_results['test_mae']:,.0f} ({gen_results['test_mape']:.1f}%)")
    print(f"📊 Model explains {fit_results['cv_r2_mean']:.1%} of price variation")

if __name__ == "__main__":
    main()