# --- Import Libraries ---
import numpy as np
import random
import os
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import time
from time import perf_counter
from io import BytesIO
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam

# --- Set Global Seed for Reproducibility ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

# --- Helper Functions ---
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
    if model_type == 'nn':
        return model.count_params() * 4
    else:
        buffer = BytesIO()
        joblib.dump(model, buffer)
        return buffer.getbuffer().nbytes

def calculate_accuracy(y_true_log, y_pred_log, model_type='default'):
    """Calculate accuracy with ±10% tolerance (±8% for tree models)"""
    y_true_exp = np.expm1(y_true_log)
    y_pred_exp = np.expm1(y_pred_log)
    
    # Stricter threshold for tree models
    threshold = 0.05 if model_type in ['xgb', 'lgbm'] else 0.10
    
    return 100 * np.mean(np.abs(y_pred_exp - y_true_exp) / y_true_exp < threshold)

def smape(y_true, y_pred):
    epsilon = 1e-10
    return 100.0 * np.mean(2.0 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + epsilon))

def evaluate_model(y_true, y_pred, dataset_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    mape = (abs((y_true - y_pred) / y_true).mean()) * 100
    smape_score = smape(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    accuracy_10 = np.mean(np.abs(y_pred - y_true) / y_true < 0.10) * 100
    print(f"\n{dataset_name} Evaluation:")
    print(f"  MSE: {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  MAPE: {mape:.4f}%")
    print(f"  sMAPE: {smape_score:.4f}%")
    print(f"  R² Score: {r2:.4f}")
    print(f"  Accuracy (±10%): {accuracy_10:.2f}%")

def scale_value(value, min_val, max_val, direction=1):
    """Scale a value to 0-1 range based on min/max and direction"""
    if direction == 1:  # Higher is better
        scaled = (value - min_val) / (max_val - min_val)
    else:  # Lower is better (inverted)
        scaled = (max_val - value) / (max_val - min_val)
    return max(0, min(1, scaled))  # Clamp to 0-1 range

# --- Load Data ---
csv_file = "FYP_Dataset_Area.csv"
df = pd.read_csv(csv_file)

# Remove extreme outliers (MBIST Area below 10 is likely unrealistic)
df = df[df["MBIST Area"] >= 10].copy()

# --- Feature Engineering ---
df["Included"] = df["No. of SRAM"] + df["No. of RF"]
df["Frac_Included"] = df["Included"] / df["Total Memories"].replace(0, 1)
df["Inc_per_Domain"] = df["Included"] / df["No. of Clock Domains"].replace(0, 1)
df["SRAM_or_RF"] = np.where(df["No. of SRAM"] > 0, 1, 0)  
df["Total_Bits"] = df["Size of Address"] * df["Data Width"]
df["Excl_Frac"] = df["Excluded Memories"] / df["Total Memories"].replace(0, 1)
df["Bits_per_Memory"] = df["Total_Bits"] / df["Total Memories"].replace(0, 1)

# --- Log-transform the Target ---
df["MBIST Area Log"] = np.log1p(df["MBIST Area"])

# --- Define Features and Target ---
features = ["No. of SRAM", "No. of RF", "Total Memories", "No. of Clock Domains",
            "Size of Address", "Data Width", "Frac_Included", "Inc_per_Domain", 
            "SRAM_or_RF", "Total_Bits", "Excl_Frac", "Bits_per_Memory"]
raw_features = ["No. of SRAM", "No. of RF", "Total Memories", "Excluded Memories", 
                "No. of Clock Domains", "Size of Address", "Data Width"]
target = "MBIST Area Log"

# --- Feature Scaling ---
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

# --- Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(df[features], df[target], test_size=0.2, random_state=SEED)
y_train_exp = np.expm1(y_train)
y_test_exp = np.expm1(y_test)

# --- Polynomial Feature Expansion ---
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# --- Create metrics dictionary ---
model_metrics = {
    'XGBoost': {'train_time': 0, 'inference_time': 0, 'complexity': 0, 'accuracy': 0, 'model_size': 0},
    'LightGBM': {'train_time': 0, 'inference_time': 0, 'complexity': 0, 'accuracy': 0, 'model_size': 0},
    'Neural Network': {'train_time': 0, 'inference_time': 0, 'complexity': 0, 'accuracy': 0, 'model_size': 0},
    'Stacked Ensemble': {'train_time': 0, 'inference_time': 0, 'complexity': 0, 'accuracy': 0, 'model_size': 0}
}

# --- Load Pretrained Models if Available ---
def load_model(model_name):
    try:
        return joblib.load(f"{model_name}.joblib")
    except FileNotFoundError:
        return None

xgb_model = load_model('xgb_model')
lgbm_model = load_model('lgbm_model')
nn_model = load_model('nn_model')
meta_model = load_model('meta_model')

if not xgb_model or not lgbm_model or not nn_model or not meta_model:
    # --- Model 1: XGBoost ---
    print("\nTraining XGBoost...")
    start = perf_counter()
    xgb_model = XGBRegressor(
        max_depth=7, learning_rate=0.0135, n_estimators=850, subsample=0.92,
        colsample_bytree=0.87, reg_alpha=0.15, reg_lambda=0.35, objective='reg:squarederror',
        random_state=SEED, n_jobs=-1
    )
    xgb_model.fit(X_train_poly, y_train, eval_set=[(X_train_poly, y_train), (X_test_poly, y_test)], verbose=False)
    joblib.dump(xgb_model, 'xgb_model.joblib')

    # --- Model 2: LightGBM ---
    print("\nTraining LightGBM...")
    start = perf_counter()
    lgbm_model = LGBMRegressor(
        max_depth=7, learning_rate=0.013, n_estimators=850, subsample=0.92,
        colsample_bytree=0.87, reg_alpha=0.15, reg_lambda=0.35, objective='regression',
        random_state=SEED, n_jobs=-1
    )
    lgbm_model.fit(X_train_poly, y_train)
    joblib.dump(lgbm_model, 'lgbm_model.joblib')

    # --- Model 3: Neural Network ---
    def build_nn(input_dim):
        model = Sequential([
            Dense(512, activation='swish', input_shape=(input_dim,)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(256, activation='swish'),
            BatchNormalization(),
            Dense(128, activation='swish'),
            Dropout(0.2),
            Dense(1, activation='linear')
        ])
        model.compile(loss='huber', optimizer=Adam(learning_rate=0.0005, decay=1e-4))
        return model

    print("\nTraining Neural Network...")
    nn_model = build_nn(X_train_poly.shape[1])
    start = perf_counter()
    nn_model.fit(X_train_poly, y_train, validation_split=0.2, epochs=300, batch_size=128, verbose=1)
    joblib.dump(nn_model, 'nn_model.joblib')

    # --- Train Stacked Ensemble ---
    print("\nTraining Stacked Ensemble...")
    train_xgb = xgb_model.predict(X_train_poly)
    train_lgbm = lgbm_model.predict(X_train_poly)
    train_nn = nn_model.predict(X_train_poly).flatten()

    X_train_meta = np.column_stack((train_xgb, train_lgbm, train_nn))  # Stacking the predictions

    meta_model = GradientBoostingRegressor(
        loss='quantile', alpha=0.5, n_estimators=200
    )
    start = perf_counter()
    meta_model.fit(X_train_meta, y_train)
    joblib.dump(meta_model, 'meta_model.joblib')

# --- Full Inference for the Test Set ---
test_xgb = xgb_model.predict(X_test_poly)
test_lgbm = lgbm_model.predict(X_test_poly)
test_nn = nn_model.predict(X_test_poly).flatten()

# Stack test predictions for meta-model
X_test_meta = np.column_stack((test_xgb, test_lgbm, test_nn))

# Get final predictions from the meta-model
y_test_pred = meta_model.predict(X_test_meta)

# Apply inverse log transformation to get the actual predictions
y_test_pred_exp = np.expm1(y_test_pred)  # Convert log-transformed predictions to original scale

# --- Evaluate the Model ---
evaluate_model(y_test_exp, y_test_pred_exp, "Test Set")

# --- Print Model Metrics ---
def print_model_metrics(metrics_dict):
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

# --- Radar Plot Function with Proper Scaling ---
def plot_model_radar(metrics_dict):
    categories = ['Training Time', 'Inference Time', 'Model Complexity', 'Accuracy', 'Model Size']
    models = list(metrics_dict.keys())
    
    # Print metrics before plotting
    print_model_metrics(metrics_dict)
    
    # Prepare data for radar plot with proper scaling
    scaled_data = {}
    original_ranges = {}
    
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
    
    # Print scaling information
    print("\n" + "="*80)
    print("METRIC SCALING INFORMATION")
    print("="*80)
    print("{:<20} {:<15} {:<15} {:<15} {:<15}".format(
        "Metric", "Actual Min", "Actual Max", "Scaled Min", "Scaled Max"
    ))
    print("-"*80)
    for category in categories:
        print("{:<20} {:<15.4f} {:<15.4f} {:<15.4f} {:<15.4f}".format(
            category,
            ranges[category]['actual_min'],
            ranges[category]['actual_max'],
            0 if ranges[category]['direction'] == 1 else 1,
            1 if ranges[category]['direction'] == 1 else 0
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
        # Scale each metric
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
    
    # Create radar plot with enhanced styling
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
    colors = ["#FF0000", "#8000FF", "#FF8400", "#06A600"]
    line_styles = ['-', '-', '-', '-']
    markers = ['o', 'o', 'o', 'o']
    
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
    plt.title('Trade-offs in Performance and Efficiency of MBIST Area Model', 
             size=16, pad=25, fontweight='bold', color='#2E4053')
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), 
              fontsize=10, framealpha=1, edgecolor='#2E4053')
    
    plt.tight_layout()
    plt.savefig('mbist_area_radar_plot.png', dpi=300)
    plt.show()

# --- Command-line Interface ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MBIST Area Prediction Model')
    parser.add_argument('-p', '--predict', action='store_true',
                        help='Enable interactive prediction mode')
    args = parser.parse_args()
    
    if args.predict:
        run_interactive_prediction()
    else:
        # Plot radar after training
        plot_model_radar(model_metrics)
