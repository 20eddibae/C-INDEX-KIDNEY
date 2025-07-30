import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load KiTS23 results and KiTS19 data, then merge them."""
    
    # Load KiTS23 results
    kits23_df = pd.read_csv('results.csv')
    print(f"Loaded {len(kits23_df)} KiTS23 cases")
    
    # Load KiTS19 data
    kits19_df = pd.read_csv('C-Index KiTS19.csv')
    print(f"Loaded {len(kits19_df)} KiTS19 cases")
    
    # Clean up KiTS19 data - remove empty rows and fix column names
    kits19_df = kits19_df.dropna(subset=['Manual C Index'])
    kits19_df = kits19_df[kits19_df['Manual C Index'] != '']
    
    # Convert case_id to match format
    kits19_df['case_id'] = kits19_df.iloc[:, 0]  # First column contains case IDs
    
    # Select relevant columns from KiTS19
    kits19_clean = kits19_df[['case_id', 'Manual C Index', 'Manual Total Distance (mm)', 'Manual Max Tumor Radius (mm)']].copy()
    kits19_clean.columns = ['case_id', 'kits19_manual_c_index', 'kits19_distance_mm', 'kits19_tumor_radius_mm']
    
    # Convert to numeric
    for col in ['kits19_manual_c_index', 'kits19_distance_mm', 'kits19_tumor_radius_mm']:
        kits19_clean[col] = pd.to_numeric(kits19_clean[col], errors='coerce')
    
    # Select relevant columns from KiTS23
    kits23_clean = kits23_df[['case_id', 'c_index', 'distance_mm', 'kidney_radius_mm']].copy()
    kits23_clean.columns = ['case_id', 'kits23_c_index', 'kits23_distance_mm', 'kits23_kidney_radius_mm']
    
    # Merge datasets on case_id
    merged_df = pd.merge(kits23_clean, kits19_clean, on='case_id', how='inner')
    
    print(f"After merging: {len(merged_df)} cases with both KiTS23 and KiTS19 data")
    
    return merged_df

def perform_regression_analysis(df):
    """Perform least squares regression analysis."""
    
    # Remove any rows with NaN values
    df_clean = df.dropna()
    print(f"After removing NaN values: {len(df_clean)} cases")
    
    # Prepare data for regression
    X = df_clean['kits23_c_index'].values.reshape(-1, 1)
    y = df_clean['kits19_manual_c_index'].values
    
    # Perform linear regression
    reg = LinearRegression()
    reg.fit(X, y)
    
    # Make predictions
    y_pred = reg.predict(X)
    
    # Calculate metrics
    r2 = r2_score(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, y_pred)
    
    # Calculate correlation coefficient
    correlation = np.corrcoef(X.flatten(), y)[0, 1]
    
    # Perform t-test for correlation
    t_stat, p_value = stats.pearsonr(X.flatten(), y)
    
    return {
        'slope': reg.coef_[0],
        'intercept': reg.intercept_,
        'r2': r2,
        'correlation': correlation,
        'p_value': p_value,
        'rmse': rmse,
        'mae': mae,
        'X': X,
        'y': y,
        'y_pred': y_pred,
        'df_clean': df_clean
    }

def create_visualizations(results, df):
    """Create comprehensive visualizations."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('C-Index Regression Analysis: KiTS23 vs KiTS19 Manual', fontsize=16, fontweight='bold')
    
    # 1. Scatter plot with regression line
    axes[0, 0].scatter(results['X'], results['y'], alpha=0.6, color='blue')
    axes[0, 0].plot(results['X'], results['y_pred'], color='red', linewidth=2, label=f'R² = {results["r2"]:.3f}')
    axes[0, 0].plot([0, max(results['X'])], [0, max(results['X'])], 'k--', alpha=0.5, label='y=x')
    axes[0, 0].set_xlabel('KiTS23 C-Index')
    axes[0, 0].set_ylabel('KiTS19 Manual C-Index')
    axes[0, 0].set_title('Regression Analysis')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Residuals plot
    residuals = results['y'] - results['y_pred']
    axes[0, 1].scatter(results['y_pred'], residuals, alpha=0.6, color='green')
    axes[0, 1].axhline(y=0, color='red', linestyle='--')
    axes[0, 1].set_xlabel('Predicted C-Index')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('Residuals Plot')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Bland-Altman plot
    mean_values = (results['X'].flatten() + results['y']) / 2
    differences = results['X'].flatten() - results['y']
    mean_diff = np.mean(differences)
    std_diff = np.std(differences)
    
    axes[0, 2].scatter(mean_values, differences, alpha=0.6, color='purple')
    axes[0, 2].axhline(y=mean_diff, color='red', linestyle='-', label=f'Mean: {mean_diff:.3f}')
    axes[0, 2].axhline(y=mean_diff + 1.96*std_diff, color='red', linestyle='--', label='+1.96 SD')
    axes[0, 2].axhline(y=mean_diff - 1.96*std_diff, color='red', linestyle='--', label='-1.96 SD')
    axes[0, 2].set_xlabel('Mean of KiTS23 and KiTS19 C-Index')
    axes[0, 2].set_ylabel('Difference (KiTS23 - KiTS19)')
    axes[0, 2].set_title('Bland-Altman Plot')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Distribution comparison
    axes[1, 0].hist(results['X'].flatten(), alpha=0.7, label='KiTS23', bins=20, color='blue')
    axes[1, 0].hist(results['y'], alpha=0.7, label='KiTS19 Manual', bins=20, color='red')
    axes[1, 0].set_xlabel('C-Index Value')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution Comparison')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Distance comparison
    axes[1, 1].scatter(df['kits23_distance_mm'], df['kits19_distance_mm'], alpha=0.6, color='orange')
    max_dist = max(df['kits23_distance_mm'].max(), df['kits19_distance_mm'].max())
    axes[1, 1].plot([0, max_dist], [0, max_dist], 'k--', alpha=0.5)
    axes[1, 1].set_xlabel('KiTS23 Distance (mm)')
    axes[1, 1].set_ylabel('KiTS19 Distance (mm)')
    axes[1, 1].set_title('Distance Comparison')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Radius comparison
    axes[1, 2].scatter(df['kits23_kidney_radius_mm'], df['kits19_tumor_radius_mm'], alpha=0.6, color='brown')
    max_radius = max(df['kits23_kidney_radius_mm'].max(), df['kits19_tumor_radius_mm'].max())
    axes[1, 2].plot([0, max_radius], [0, max_radius], 'k--', alpha=0.5)
    axes[1, 2].set_xlabel('KiTS23 Kidney Radius (mm)')
    axes[1, 2].set_ylabel('KiTS19 Tumor Radius (mm)')
    axes[1, 2].set_title('Radius Comparison')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('c_index_regression_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_statistical_summary(results, df):
    """Print comprehensive statistical summary."""
    
    print("\n" + "="*80)
    print("C-INDEX REGRESSION ANALYSIS: KiTS23 vs KiTS19 MANUAL")
    print("="*80)
    
    print(f"\nDataset Information:")
    print(f"  Total cases analyzed: {len(results['df_clean'])}")
    print(f"  KiTS23 C-index range: {results['X'].min():.3f} - {results['X'].max():.3f}")
    print(f"  KiTS19 C-index range: {results['y'].min():.3f} - {results['y'].max():.3f}")
    
    print(f"\nRegression Results:")
    print(f"  Equation: KiTS19_C_index = {results['slope']:.3f} × KiTS23_C_index + {results['intercept']:.3f}")
    print(f"  R² (Coefficient of determination): {results['r2']:.3f}")
    print(f"  Correlation coefficient (r): {results['correlation']:.3f}")
    print(f"  P-value: {results['p_value']:.6f}")
    print(f"  RMSE (Root Mean Square Error): {results['rmse']:.3f}")
    print(f"  MAE (Mean Absolute Error): {results['mae']:.3f}")
    
    print(f"\nStatistical Significance:")
    if results['p_value'] < 0.001:
        significance = "*** (p < 0.001)"
    elif results['p_value'] < 0.01:
        significance = "** (p < 0.01)"
    elif results['p_value'] < 0.05:
        significance = "* (p < 0.05)"
    else:
        significance = "Not significant (p ≥ 0.05)"
    
    print(f"  Correlation significance: {significance}")
    
    print(f"\nCorrelation Strength:")
    if abs(results['correlation']) >= 0.9:
        strength = "Very Strong"
    elif abs(results['correlation']) >= 0.7:
        strength = "Strong"
    elif abs(results['correlation']) >= 0.5:
        strength = "Moderate"
    elif abs(results['correlation']) >= 0.3:
        strength = "Weak"
    else:
        strength = "Very Weak"
    
    print(f"  Correlation strength: {strength}")
    
    # Additional statistics
    print(f"\nDescriptive Statistics:")
    print(f"  KiTS23 C-index - Mean: {np.mean(results['X']):.3f}, Std: {np.std(results['X']):.3f}")
    print(f"  KiTS19 C-index - Mean: {np.mean(results['y']):.3f}, Std: {np.std(results['y']):.3f}")
    
    # Calculate bias and limits of agreement for Bland-Altman
    differences = results['X'].flatten() - results['y']
    mean_diff = np.mean(differences)
    std_diff = np.std(differences)
    print(f"\nBland-Altman Analysis:")
    print(f"  Mean difference (bias): {mean_diff:.3f}")
    print(f"  Standard deviation of differences: {std_diff:.3f}")
    print(f"  95% Limits of Agreement: [{mean_diff - 1.96*std_diff:.3f}, {mean_diff + 1.96*std_diff:.3f}]")

def save_detailed_results(results, df):
    """Save detailed results to CSV."""
    
    # Create detailed results dataframe
    detailed_df = results['df_clean'].copy()
    detailed_df['predicted_kits19'] = results['y_pred']
    detailed_df['residuals'] = results['y'] - results['y_pred']
    detailed_df['absolute_error'] = np.abs(detailed_df['residuals'])
    detailed_df['percentage_error'] = (detailed_df['absolute_error'] / detailed_df['kits19_manual_c_index']) * 100
    
    # Add regression statistics
    detailed_df['regression_equation'] = f"y = {results['slope']:.3f}x + {results['intercept']:.3f}"
    detailed_df['r_squared'] = results['r2']
    detailed_df['correlation'] = results['correlation']
    detailed_df['p_value'] = results['p_value']
    
    # Save to CSV
    detailed_df.to_csv('detailed_regression_results.csv', index=False)
    print(f"\nDetailed results saved to 'detailed_regression_results.csv'")

def main():
    """Main function to run the complete analysis."""
    
    print("Loading and preparing data...")
    df = load_and_prepare_data()
    
    print("Performing regression analysis...")
    results = perform_regression_analysis(df)
    
    print("Creating visualizations...")
    create_visualizations(results, df)
    
    print("Generating statistical summary...")
    print_statistical_summary(results, df)
    
    print("Saving detailed results...")
    save_detailed_results(results, df)
    
    print("\nAnalysis complete! Check the generated files:")
    print("  - c_index_regression_analysis.png (visualizations)")
    print("  - detailed_regression_results.csv (detailed data)")

if __name__ == "__main__":
    main() 