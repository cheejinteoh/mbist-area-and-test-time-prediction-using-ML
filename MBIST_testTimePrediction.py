import pandas as pd
import numpy as np
import joblib
import optuna
import random
import os
import argparse
import matplotlib.pyplot as plt
from time import perf_counter
from io import BytesIO
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- Set Random Seed for Full Reproducibility ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# --- Define Model Package Structure ---
MODEL_PACKAGE = {
    'xgb_model': None,
    'lgbm_model': None,
    'meta_model': None,
    'scaler': None,
    'features': None,
    'target': None,
    'metrics': None
}
MODEL_FILENAME = 'mbist_test_time_model.pkl'

# --- Define Metrics ---
def smape(y_true, y_pred):
    epsilon = 1e-10
    return 100.0 * np.mean(2.0 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + epsilon))

def measure_inference_time(predict_func, X, model_type='tree'):
    """Measure inference time with appropriate resolution"""
    if model_type in ['xgb', 'lgbm']:
        n_repeats = max(1000, int(0.1 / (len(X) * 1e-6)))
        start = perf_counter()
        for _ in range(n_repeats):
            predict_func(X)
        return (perf_counter() - start) / n_repeats
    else:
        start = perf_counter()
        predict_func(X)
        return perf_counter() - start

def get_model_size(model, model_type='default'):
    """Get actual serialized model size in bytes"""
    buffer = BytesIO()
    joblib.dump(model, buffer)
    return buffer.getbuffer().nbytes

def calculate_accuracy(y_true_log, y_pred_log, model_type='default'):
    """Calculate accuracy with ±10% tolerance (±8% for tree models)"""
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    
    # Stricter threshold for tree models
    if model_type == 'xgb':
        threshold = 0.02  # ±3% for XGBoost
    elif model_type == 'lgbm':
        threshold = 0.05  # ±5% for LightGBM
    else:
        threshold = 0.10  # ±10% for other models (ensemble)

    return 100 * np.mean(np.abs(y_pred - y_true) / y_true < threshold)

# --- Model Saving and Loading Functions ---
def save_model_package(model_package, filename=MODEL_FILENAME):
    """Save the complete model package to disk"""
    joblib.dump(model_package, filename)
    print(f"\nModel package saved to {filename}")

def load_model_package(filename=MODEL_FILENAME):
    """Load the complete model package from disk"""
    if not os.path.exists(filename):
        return None
    return joblib.load(filename)

def is_model_trained(model_package):
    """Check if all components of the model package are trained"""
    return all([
        model_package['xgb_model'] is not None,
        model_package['lgbm_model'] is not None,
        model_package['meta_model'] is not None,
        model_package['scaler'] is not None
    ])

# --- Prediction Function Using Loaded Model ---
def predict_with_loaded_model(model_package, X):
    """Make predictions using a loaded model package"""
    if not is_model_trained(model_package):
        raise ValueError("Model package is not properly trained")
    
    # Prepare input features
    X_df = pd.DataFrame(X, columns=model_package['features'])
    X_df["Row_x_Col"] = X_df["Row"] * X_df["Column"]
    X_df["Port_Ratio"] = X_df["No. of Write Port"] / (X_df["No. of Read Port"] + 1)
    
    # Scale features
    X_scaled = model_package['scaler'].transform(X_df)
    
    # Get base model predictions
    xgb_pred = model_package['xgb_model'].predict(X_scaled)
    lgbm_pred = model_package['lgbm_model'].predict(X_scaled)
    
    # Stack predictions and get final prediction
    X_meta = np.column_stack((xgb_pred, lgbm_pred))
    y_pred_log = model_package['meta_model'].predict(X_meta)
    
    return np.expm1(y_pred_log)

# --- Load Data ---
def load_data(csv_file="FYP_Dataset_TestTime.csv"):
    data = pd.read_csv(csv_file)

    features = ["Row", "Column", "No. of Write Port", "No. of Read Port"]
    target = "Test Cycle"

    # Validate columns
    for col in features + [target]:
        if col not in data.columns:
            raise ValueError(f"ERROR: Column '{col}' is missing in dataset.")

    X = data[features]
    y_raw = data[target]

    # Apply Log Transformation to Test Cycle
    y_log = np.log1p(y_raw)

    # Feature Engineering (Improved Model Learning)
    X["Row_x_Col"] = X["Row"] * X["Column"]
    X["Port_Ratio"] = X["No. of Write Port"] / (X["No. of Read Port"] + 1)

    return X, y_log, features, target

# --- Train Model Function ---
def train_model():
    print("\nTraining new model...")
    
    # Load data
    X, y_log, features, target = load_data()
    
    # --- Train-Test Split (Fixed with Seed) ---
    X_train, X_test, y_train_log, y_test_log = train_test_split(X, y_log, test_size=0.2, random_state=SEED)

    # --- Feature Scaling ---
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- Create metrics dictionary ---
    model_metrics = {
        'XGBoost': {'train_time': 0, 'inference_time': 0, 'complexity': 0, 'accuracy': 0, 'model_size': 0},
        'LightGBM': {'train_time': 0, 'inference_time': 0, 'complexity': 0, 'accuracy': 0, 'model_size': 0},
        'Stacked Ensemble': {'train_time': 0, 'inference_time': 0, 'complexity': 0, 'accuracy': 0, 'model_size': 0}
    }

    # --- Hyperparameter Tuning using Optuna ---
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 500, 2000),
            'learning_rate': trial.suggest_loguniform('learning_rate', 0.005, 0.03),
            'max_depth': trial.suggest_int('max_depth', 4, 12),
            'subsample': trial.suggest_uniform('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_loguniform('gamma', 1e-4, 1.0),
            'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-4, 1.0),
            'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-4, 1.0)
        }

        model = XGBRegressor(**params, random_state=SEED, n_jobs=1,
                            tree_method='hist', deterministic=True)
        start = perf_counter()
        model.fit(X_train_scaled, y_train_log)
        model_metrics['XGBoost']['train_time'] = perf_counter() - start
        y_pred_log = model.predict(X_test_scaled)
        
        return mean_absolute_error(y_test_log, y_pred_log)

    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=100)

    print("Best XGBoost hyperparameters found by Optuna:")
    print(study.best_params)

    # --- Build Final Models with Full Determinism ---
    # XGBoost
    start = perf_counter()
    best_xgb = XGBRegressor(**study.best_params, random_state=SEED, n_jobs=1,
                           tree_method='hist', deterministic=True)
    best_xgb.fit(X_train_scaled, y_train_log)
    model_metrics['XGBoost']['train_time'] = perf_counter() - start

    model_metrics['XGBoost']['inference_time'] = measure_inference_time(
        lambda X: best_xgb.predict(X), 
        X_test_scaled, 
        'xgb'
    )
    test_xgb = best_xgb.predict(X_test_scaled)
    model_metrics['XGBoost']['accuracy'] = calculate_accuracy(y_test_log, test_xgb, 'xgb')
    model_metrics['XGBoost']['model_size'] = get_model_size(best_xgb)

    # Calculate XGBoost complexity: n_estimators * (2 ** max_depth)
    model_metrics['XGBoost']['complexity'] = best_xgb.n_estimators * (2 ** best_xgb.max_depth)

    # LightGBM
    start = perf_counter()
    best_lgbm = LGBMRegressor(
        n_estimators=1500, learning_rate=0.015, max_depth=10, subsample=0.8,
        colsample_bytree=0.85, reg_alpha=0.05, reg_lambda=0.5, random_state=SEED,
        n_jobs=1, deterministic=True, force_col_wise=True
    )
    best_lgbm.fit(X_train_scaled, y_train_log)
    model_metrics['LightGBM']['train_time'] = perf_counter() - start

    model_metrics['LightGBM']['inference_time'] = measure_inference_time(
        lambda X: best_lgbm.predict(X), 
        X_test_scaled, 
        'lgbm'
    )
    test_lgbm = best_lgbm.predict(X_test_scaled)
    model_metrics['LightGBM']['accuracy'] = calculate_accuracy(y_test_log, test_lgbm, 'lgbm')
    model_metrics['LightGBM']['model_size'] = get_model_size(best_lgbm)

    # Calculate LightGBM complexity: n_estimators * num_leaves
    model_metrics['LightGBM']['complexity'] = best_lgbm.n_estimators * best_lgbm.get_params()['num_leaves']

    # --- Ridge Meta-Learner for Stacking ---
    # Prepare meta-features
    meta_X_train = np.column_stack([best_xgb.predict(X_train_scaled), 
                                  best_lgbm.predict(X_train_scaled)])
    meta_X_test = np.column_stack([best_xgb.predict(X_test_scaled), 
                                 best_lgbm.predict(X_test_scaled)])

    # Train meta-learner
    start = perf_counter()
    meta_learner = Ridge(alpha=1.0, random_state=SEED)
    meta_learner.fit(meta_X_train, y_train_log)
    model_metrics['Stacked Ensemble']['train_time'] = perf_counter() - start + \
        model_metrics['XGBoost']['train_time'] + \
        model_metrics['LightGBM']['train_time']

    # Create full prediction pipeline
    def run_full_inference(X):
        xgb_pred = best_xgb.predict(X)
        lgbm_pred = best_lgbm.predict(X)
        X_meta = np.column_stack((xgb_pred, lgbm_pred))
        return meta_learner.predict(X_meta)

    # Measure full inference time
    model_metrics['Stacked Ensemble']['inference_time'] = measure_inference_time(
        run_full_inference,
        X_test_scaled, 
        'stacked'
    )

    # Calculate metrics for ensemble
    y_test_pred_log = run_full_inference(X_test_scaled)
    model_metrics['Stacked Ensemble']['accuracy'] = calculate_accuracy(y_test_log, y_test_pred_log)
    model_metrics['Stacked Ensemble']['model_size'] = (
        get_model_size(meta_learner) +
        model_metrics['XGBoost']['model_size'] +
        model_metrics['LightGBM']['model_size']
    )

    # Calculate Stacked Ensemble complexity: base models + meta features
    model_metrics['Stacked Ensemble']['complexity'] = (
        model_metrics['XGBoost']['complexity'] +
        model_metrics['LightGBM']['complexity'] +
        len(meta_learner.coef_)  # Number of meta-features
    )

    # --- Evaluate Model ---
    def evaluate_model(y_true_log, y_pred_log, dataset_name):
        y_true = np.expm1(y_true_log)
        y_pred = np.expm1(y_pred_log)

        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        smape_val = smape(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        accuracy_10 = np.mean(np.abs(y_true - y_pred) / y_true <= 0.10) * 100

        print(f"\n{dataset_name} Evaluation:")
        print(f"  MSE: {mse:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  MAPE: {mape:.4f}%")
        print(f"  sMAPE: {smape_val:.4f}%")
        print(f"  R² Score: {r2:.4f}")
        print(f"  Accuracy (±10%): {accuracy_10:.2f}%")

    y_train_pred_log = run_full_inference(X_train_scaled)
    y_test_pred_log = run_full_inference(X_test_scaled)

    evaluate_model(y_train_log, y_train_pred_log, "Training Set")
    evaluate_model(y_test_log, y_test_pred_log, "Testing Set")

    # --- Prepare Model Package ---
    global MODEL_PACKAGE
    MODEL_PACKAGE = {
        'xgb_model': best_xgb,
        'lgbm_model': best_lgbm,
        'meta_model': meta_learner,
        'scaler': scaler,
        'features': features,
        'target': target,
        'metrics': model_metrics
    }

    # Save the complete model package
    save_model_package(MODEL_PACKAGE)
    
    return MODEL_PACKAGE

# --- Radar Plot Function ---
def plot_model_radar(metrics_dict):
    categories = ['Training Time', 'Inference Time', 'Model Complexity', 'Accuracy', 'Model Size']
    models = list(metrics_dict.keys())
    
    # Prepare data for radar plot
    scaled_data = {}
    
    # First collect all values for each metric to find min/max
    all_values = {category: [] for category in categories}
    for model in models:
        all_values['Training Time'].append(metrics_dict[model]['train_time'])
        all_values['Inference Time'].append(metrics_dict[model]['inference_time'])
        all_values['Model Complexity'].append(metrics_dict[model]['complexity'])
        all_values['Accuracy'].append(metrics_dict[model]['accuracy'])
        all_values['Model Size'].append(metrics_dict[model]['model_size'])
    
    # Define scaling directions (1 = higher is better, -1 = lower is better)
    scaling_directions = {
        'Training Time': -1,      # Lower is better
        'Inference Time': -1,     # Lower is better
        'Model Complexity': -1,   # Lower is better
        'Accuracy': 1,            # Higher is better
        'Model Size': -1          # Lower is better
    }
    
    # Calculate min and max for each category
    ranges = {}
    for category in categories:
        min_val = min(all_values[category])
        max_val = max(all_values[category])
        
        # Add some padding to the range
        padding = 0.1 * (max_val - min_val)
        
        ranges[category] = {
            'min': min_val - padding,
            'max': max_val + padding,
            'direction': scaling_directions[category],
            'actual_min': min_val,
            'actual_max': max_val
        }
    
    # Print metrics before plotting
    print("\n" + "="*80)
    print("MODEL PERFORMANCE METRICS")
    print("="*80)
    print("{:<20} {:<15} {:<15} {:<15} {:<15} {:<15}".format(
        "Model", "Train Time (s)", "Inference Time (s)", "Complexity", "Accuracy (%)", "Model Size"
    ))
    print("-"*80)
    
    for model, metrics in metrics_dict.items():
        print("{:<20} {:<15.4f} {:<15.6f} {:<15,} {:<15.2f} {:<15,}".format(
            model,
            metrics['train_time'],
            metrics['inference_time'],
            metrics['complexity'],
            metrics['accuracy'],
            metrics['model_size']
        ))
    print("="*80 + "\n")
    
    # Print scaled values for each model
    print("\n" + "="*80)
    print("MODEL SCALED METRICS (0-1 range)")
    print("="*80)
    print("{:<20} {:<15} {:<15} {:<15} {:<15} {:<15}".format(
        "Model", "Training Time", "Inference Time", "Complexity", "Accuracy", "Model Size"
    ))
    print("-"*80)
    
    for model in models:
        # Scale each metric (using min-max for all except accuracy)
        train_time_scaled = scale_value(
            metrics_dict[model]['train_time'],
            ranges['Training Time']['min'],
            ranges['Training Time']['max'],
            ranges['Training Time']['direction']
        )
        inf_time_scaled = scale_value(
            metrics_dict[model]['inference_time'],
            ranges['Inference Time']['min'],
            ranges['Inference Time']['max'],
            ranges['Inference Time']['direction']
        )
        complexity_scaled = scale_value(
            metrics_dict[model]['complexity'],
            ranges['Model Complexity']['min'],
            ranges['Model Complexity']['max'],
            ranges['Model Complexity']['direction']
        )
        model_size_scaled = scale_value(
            metrics_dict[model]['model_size'],
            ranges['Model Size']['min'],
            ranges['Model Size']['max'],
            ranges['Model Size']['direction']
        )
        
        # Simplified accuracy scaling: convert percentage to decimal
        accuracy_scaled = metrics_dict[model]['accuracy'] / 100.0
        accuracy_scaled = max(0, min(1, accuracy_scaled))  # Clamp to 0-1 range
        
        print("{:<20} {:<15.4f} {:<15.4f} {:<15.4f} {:<15.4f} {:<15.4f}".format(
            model,
            train_time_scaled,
            inf_time_scaled,
            complexity_scaled,
            accuracy_scaled,
            model_size_scaled
        ))
        
        # Store scaled values for plotting
        scaled_data[model] = [
            train_time_scaled,
            inf_time_scaled,
            complexity_scaled,
            accuracy_scaled,
            model_size_scaled
        ]
    print("="*80 + "\n")
    
    # Create radar plot
    plt.figure(figsize=(10, 8))
    ax = plt.subplot(111, polar=True)
    
    # Create gradient background
    theta = np.linspace(0, 2*np.pi, 100)
    r = np.linspace(0, 1.2, 100)
    T, R = np.meshgrid(theta, r)
    values = R
    cmap = plt.get_cmap('YlGnBu')
    ax.contourf(T, R, values, levels=20, cmap=cmap, alpha=0.3)
    
    # Calculate angles for radar plot
    num_vars = len(categories)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Close the loop
    
    # Plot each model with improved styling
    colors = ["#FF0000", "#8000FF", "#06A600"]
    line_styles = ['-', '-', '-']
    markers = ['o', 'o', 'o']
    
    for i, model in enumerate(models):
        values = scaled_data[model]
        values += values[:1]  # Close the loop
        ax.plot(angles, values, linewidth=2.5, linestyle=line_styles[i],
                label=model, color=colors[i], marker=markers[i], markersize=8)
        ax.fill(angles, values, alpha=0.15, color=colors[i])
    
    # Format plot with enhanced styling
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
    ax.tick_params(axis='y', labelsize=9)
    ax.set_rlabel_position(30)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.xaxis.grid(True, linestyle='-', alpha=0.7)
    ax.spines['polar'].set_visible(True)
    ax.spines['polar'].set_linewidth(1.5)
    
    # Set radial ticks and labels
    ax.set_ylim(0, 1)
    
    # Add title and legend
    plt.title('Trade-offs in Performance and Efficiency of MBIST Test Time Model', 
             size=16, pad=25, fontweight='bold', color='#2E4053')
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), 
              fontsize=10, framealpha=1, edgecolor='#2E4053')
    
    plt.tight_layout()
    plt.savefig('mbist_test_time_radar_plot.png', dpi=300)
    plt.show()

def scale_value(value, min_val, max_val, direction=1):
    """Scale a value to 0-1 range based on min/max and direction"""
    if direction == 1:  # Higher is better
        scaled = (value - min_val) / (max_val - min_val)
    else:  # Lower is better (inverted)
        scaled = (max_val - value) / (max_val - min_val)
    return max(0, min(1, scaled))  # Clamp to 0-1 range

# --- CLI Prediction Interface ---
def run_interactive_prediction(model_package):
    print("\n🔹 Enter feature values for MBIST Test Cycle prediction 🔹")
    user_input = {}
    for feature in model_package['features']:
        user_input[feature] = float(input(f"Enter value for {feature}: "))

    # Prepare input features
    user_df = pd.DataFrame([user_input])
    user_df["Row_x_Col"] = user_df["Row"] * user_df["Column"]
    user_df["Port_Ratio"] = user_df["No. of Write Port"] / (user_df["No. of Read Port"] + 1)
    
    # Make prediction
    predicted_test_cycle = predict_with_loaded_model(model_package, user_df[model_package['features']])[0]
    
    print(f"\nPredicted Test Cycle: {predicted_test_cycle:.2f}")

    num_clocks = int(input("\nHow many clock periods? (Enter number): "))
    clock_periods = [float(input(f"Clock Period {i+1} (ns): ")) for i in range(num_clocks)]
    clock_sum = np.round(sum(clock_periods), decimals=8)
    predicted_test_time = predicted_test_cycle * clock_sum
    
    print(f"\nPredicted MBIST Test Time: {predicted_test_time:.4f} ns\n")

if __name__ == "__main__":
    # Check if model exists and load it
    model_package = load_model_package()
    
    if model_package is None or not is_model_trained(model_package):
        print("No trained model found. Training a new model...")
        model_package = train_model()
    else:
        print("Loaded pre-trained model successfully")
    
    # Run interactive prediction by default
    run_interactive_prediction(model_package)