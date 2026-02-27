import pandas as pd
import numpy as np
import optuna
import warnings
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import PartialDependenceDisplay
import xgboost as xgb
import lightgbm as lgb

# Suppress unnecessary warnings for clean terminal output
warnings.filterwarnings('ignore')

# ==========================================
# Module 1: Academic Global Visualization Setup
# ==========================================
def setup_publication_style(font_path):
    """Configure Nature/IEEE-tier chart aesthetics, binding local fonts."""
    try:
        font_manager.fontManager.addfont(font_path)
        prop = font_manager.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = prop.get_name()
        print(f"[UI Setup] Successfully mounted global academic font: {prop.get_name()}")
    except Exception as e:
        print(f"[UI Setup] Font loading failed, falling back to sans-serif. Error: {e}")
        plt.rcParams['font.family'] = 'sans-serif'
        prop = font_manager.FontProperties(family='sans-serif')

    # Core academic parameters
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'axes.titleweight': 'bold',
        'axes.labelweight': 'bold',
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        'legend.frameon': False,     
        'axes.linewidth': 1.5,       
        'xtick.major.width': 1.5,
        'ytick.major.width': 1.5,
        'xtick.direction': 'in',     
        'ytick.direction': 'in',
        'figure.dpi': 600,           
        'savefig.bbox': 'tight',     
        'pdf.fonttype': 42,          
        'ps.fonttype': 42
    })
    
    # Morandi / restrained academic color palette
    colors = ['#3C5488', '#E64B35', '#00A087', '#4DBBD5', '#F39B7F', '#8491B4', '#91D1C2', '#DC0000']
    sns.set_palette(sns.color_palette(colors))
    return prop

def save_publish_fig(fig, filename):
    """Universal function to export figures in PNG and SVG formats."""
    fig.savefig(f"{filename}.png", dpi=600, format='png', bbox_inches='tight')
    fig.savefig(f"{filename}.svg", format='svg', bbox_inches='tight')
    print(f"  -> Successfully exported: {filename}.png / .svg")

# ==========================================
# Module 2: Data Pipeline & Feature Engineering
# ==========================================
def load_and_engineer_data(file_path):
    print(f"\n[Data Pipeline] Loading data and constructing deep features...")
    df = pd.read_csv(file_path)
    
    # 1. Temporal Parsing
    df['T_timestamp'] = pd.to_datetime(df['T_timestamp'])
    df['T_year'] = df['T_timestamp'].dt.year
    df['T_month'] = df['T_timestamp'].dt.month
    df['T_day'] = df['T_timestamp'].dt.day
    df['T_dayofweek'] = df['T_timestamp'].dt.dayofweek
    df['D_weekend'] = df['T_dayofweek'].apply(lambda x: 1 if x >= 5 else 0)
    
    # 2. High-order cyclical temporal encoding
    df['T_month_sin'] = np.sin(2 * np.pi * df['T_month'] / 12)
    df['T_month_cos'] = np.cos(2 * np.pi * df['T_month'] / 12)
    df['T_dayofweek_sin'] = np.sin(2 * np.pi * df['T_dayofweek'] / 7)
    df['T_dayofweek_cos'] = np.cos(2 * np.pi * df['T_dayofweek'] / 7)
    
    # 3. Core economic indicators
    df['X_unit_price'] = df['X_revenue'] / (df['X_volume'] + 1e-5)
    df['X_unit_margin'] = df['Y_target'] / (df['X_volume'] + 1e-5)
    df['X_profit_margin'] = df['Y_target'] / (df['X_revenue'] + 1e-5)
    
    # 4. Categorical variable encoding
    cat_cols = ['Z_item_id', 'Z_category', 'Z_region']
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[f'{col}_code'] = le.fit_transform(df[col])
        label_encoders[col] = le
        
    return df, cat_cols

# ==========================================
# Module 3: AutoML & Ensemble Learning
# ==========================================
def build_and_optimize_models(X_train, y_train):
    print("\n[AutoML] Initiating Optuna optimization sequence...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    def xgb_objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 600),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'random_state': 42,
            'n_jobs': -1
        }
        model = xgb.XGBRegressor(**params)
        kf = KFold(n_splits=3, shuffle=True, random_state=42)
        scores = []
        for tr_idx, va_idx in kf.split(X_train):
            model.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx])
            preds = model.predict(X_train.iloc[va_idx])
            scores.append(np.sqrt(mean_squared_error(y_train.iloc[va_idx], preds)))
        return np.mean(scores)
    
    # Optimize XGBoost
    xgb_study = optuna.create_study(direction="minimize")
    xgb_study.optimize(xgb_objective, n_trials=15, show_progress_bar=True)
    best_xgb = xgb.XGBRegressor(**xgb_study.best_params, random_state=42, n_jobs=-1)
    
    # Independent fit for partial dependence & importance analysis
    best_xgb.fit(X_train, y_train)
    
    # Instantiate auxiliary models
    best_lgb = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=42, n_jobs=-1, verbose=-1)
    rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    
    print("\n[Ensemble] Constructing meta-learner stacking architecture...")
    estimators = [
        ('xgb', best_xgb),
        ('lgb', best_lgb),
        ('rf', rf)
    ]
    
    stacking_model = StackingRegressor(
        estimators=estimators,
        final_estimator=RidgeCV(),
        cv=5,
        n_jobs=-1
    )
    
    stacking_model.fit(X_train, y_train)
    return stacking_model, best_xgb

# ==========================================
# Module 4: Academic Plot Generation
# ==========================================
def generate_publication_plots(df, X_train, X_test, y_test, final_preds, xgb_model, features, prop):
    print("\n[Visualization] Generating publication-ready figures...")
    
    def clean_spines(ax):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    # 1. Actual vs Predicted
    fig1, ax1 = plt.subplots(figsize=(7, 7))
    sns.regplot(x=y_test, y=final_preds, ax=ax1, 
                scatter_kws={'alpha':0.6, 's':50, 'edgecolor':'white'}, 
                line_kws={'color':'#E64B35', 'linewidth':2, 'linestyle':'--'})
    min_val, max_val = min(y_test.min(), final_preds.min()), max(y_test.max(), final_preds.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'k-', lw=1.5, alpha=0.5, label='Perfect Prediction')
    ax1.set_xlabel('Observed Target ($Y$)', fontproperties=prop)
    ax1.set_ylabel('Predicted Target ($\hat{Y}$)', fontproperties=prop)
    ax1.set_title('Model Fit Assessment', fontproperties=prop, pad=15)
    r2, rmse = r2_score(y_test, final_preds), np.sqrt(mean_squared_error(y_test, final_preds))
    ax1.text(0.05, 0.95, f'$R^2 = {r2:.4f}$\n$RMSE = {rmse:.2f}$', transform=ax1.transAxes, 
             fontsize=12, va='top', bbox=dict(boxstyle='round', facecolor='#F2F2F2', edgecolor='none', alpha=0.8))
    clean_spines(ax1)
    save_publish_fig(fig1, 'Fig1_Prediction_Accuracy')
    plt.close(fig1)

    # 2. Residual Analysis
    fig2, (ax2_1, ax2_2) = plt.subplots(1, 2, figsize=(12, 5))
    residuals = y_test - final_preds
    
    ax2_1.scatter(final_preds, residuals, alpha=0.5, c='#3C5488', edgecolors='white')
    ax2_1.axhline(0, color='#E64B35', linestyle='--', lw=2)
    ax2_1.set_xlabel('Fitted Values', fontproperties=prop)
    ax2_1.set_ylabel('Residuals ($\epsilon$)', fontproperties=prop)
    ax2_1.set_title('Heteroskedasticity Diagnostic', fontproperties=prop)
    clean_spines(ax2_1)
    
    sns.kdeplot(residuals, fill=True, color='#00A087', ax=ax2_2, alpha=0.6)
    ax2_2.axvline(0, color='black', linestyle='--', lw=2)
    ax2_2.set_xlabel('Residual Error', fontproperties=prop)
    ax2_2.set_ylabel('Density', fontproperties=prop)
    ax2_2.set_title('Error Distribution', fontproperties=prop)
    clean_spines(ax2_2)
    save_publish_fig(fig2, 'Fig2_Residual_Analysis')
    plt.close(fig2)

    # 3. Feature Importance
    importances = xgb_model.feature_importances_
    indices = np.argsort(importances)[-10:]
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    ax3.barh(range(len(indices)), importances[indices], color='#4DBBD5', align='center', edgecolor='black', linewidth=1)
    ax3.set_yticks(range(len(indices)))
    ax3.set_yticklabels([features[i] for i in indices], fontproperties=prop)
    ax3.set_xlabel('Relative Importance', fontproperties=prop)
    ax3.set_title('Top 10 Feature Importances', fontproperties=prop, pad=15)
    clean_spines(ax3)
    save_publish_fig(fig3, 'Fig3_Feature_Importance')
    plt.close(fig3)

    # 4. Partial Dependence Plot
    fig4, ax4 = plt.subplots(figsize=(10, 5))
    top_features = ['X_unit_price', 'X_volume']
    PartialDependenceDisplay.from_estimator(xgb_model, X_train, features=top_features, 
                                            feature_names=features, ax=ax4,
                                            line_kw={"color": "#E64B35", "linewidth": 2.5})
    fig4.suptitle('Marginal Effects: Partial Dependence', fontweight='bold')
    plt.tight_layout()
    save_publish_fig(fig4, 'Fig4_Partial_Dependence')
    plt.close(fig4)

    # 5. Correlation Matrix
    fig5, ax5 = plt.subplots(figsize=(8, 6))
    eda_cols = ['X_volume', 'X_revenue', 'Y_target', 'X_unit_price', 'X_profit_margin', 'D_weekend']
    corr = df[eda_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1.0, vmin=-1.0, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .7}, annot=True, fmt=".2f", ax=ax5)
    ax5.set_title('Correlation Matrix', fontproperties=prop, pad=15)
    save_publish_fig(fig5, 'Fig5_Correlation_Matrix')
    plt.close(fig5)

    # 6. Violin Plot
    fig6, ax6 = plt.subplots(figsize=(10, 6))
    sns.violinplot(x='Z_region', y='Y_target', hue='Z_category', data=df, split=False, 
                   inner="quartile", palette="muted", ax=ax6)
    ax6.set_title('Target Distribution Density by Region & Category', fontproperties=prop, pad=15)
    ax6.set_xlabel('Geographical Strata (Z_region)', fontproperties=prop)
    ax6.set_ylabel('Target Variable (Y)', fontproperties=prop)
    ax6.legend(title='Category', loc='upper right', frameon=False)
    clean_spines(ax6)
    save_publish_fig(fig6, 'Fig6_Target_ViolinPlot')
    plt.close(fig6)

# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":
    print(f"{'='*50}\n Initiating Predictive Modeling Workflow\n{'='*50}")
    
    FONT_PATH = "TimesSimSunRegular.ttf"
    DATA_PATH = "data.csv"
    
    font_prop = setup_publication_style(FONT_PATH)
    
    if not os.path.exists(DATA_PATH):
        # Generate dummy data if file is missing for demonstration
        print("[Warning] Data file not found. Generating dummy dataset for execution.")
        dates = pd.date_range('2023-01-01', periods=1000)
        df = pd.DataFrame({
            'T_timestamp': dates,
            'X_volume': np.random.randint(1, 50, 1000),
            'X_revenue': np.random.uniform(10, 5000, 1000),
            'Y_target': np.random.normal(50, 20, 1000),
            'Z_item_id': np.random.choice(['Item_A', 'Item_B', 'Item_C'], 1000),
            'Z_category': np.random.choice(['Cat_1', 'Cat_2'], 1000),
            'Z_region': np.random.choice(['Reg_N', 'Reg_S', 'Reg_E', 'Reg_W'], 1000)
        })
        df.to_csv(DATA_PATH, index=False)
        
    df, cat_cols = load_and_engineer_data(DATA_PATH)
    
    feature_cols = [f'{col}_code' for col in cat_cols] + \
                   ['X_volume', 'X_revenue', 'T_year', 'T_month_sin', 'T_month_cos', 
                    'T_dayofweek_sin', 'T_dayofweek_cos', 'D_weekend', 
                    'X_unit_price', 'X_profit_margin', 'X_unit_margin']
    
    X = df[feature_cols]
    y = df['Y_target']
    
    # Fill any NaNs created by division
    X.fillna(0, inplace=True)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    stacking_model, xgb_base_model = build_and_optimize_models(X_train, y_train)
    
    final_preds = stacking_model.predict(X_test)
    print(f"\n[Evaluation] Hold-out set performance metrics:")
    print(f"  R^2 Score : {r2_score(y_test, final_preds):.4f}")
    print(f"  RMSE      : {np.sqrt(mean_squared_error(y_test, final_preds)):.4f}")
    print(f"  MAE       : {mean_absolute_error(y_test, final_preds):.4f}")
    
    generate_publication_plots(df, X_train, X_test, y_test, final_preds, 
                               xgb_base_model, feature_cols, font_prop)
    
    print(f"\n{'='*50}\n Workflow complete. High-resolution figures exported.\n{'='*50}")
