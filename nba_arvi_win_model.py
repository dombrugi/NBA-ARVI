import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

def run_arvi_model(player_stats_csv, team_wins_csv, output_csv="team_win_predictions.csv"):
    # ========================
    # STEP 1: Load & Prepare Data
    # ========================
    df = pd.read_csv(player_stats_csv)
    team_wins = pd.read_csv(team_wins_csv)

    # Ensure correct dtypes
    cols_to_numeric = ['VORP', 'MP', 'G', 'USG%', 'BPM', 'TS%']
    df[cols_to_numeric] = df[cols_to_numeric].apply(pd.to_numeric, errors='coerce').fillna(0)
    df['G'].replace(0, 1, inplace=True)
    df['MPG'] = df['MP'] / df['G']

    # ========================
    # STEP 2: Calculate ARVI
    # ========================
    team_totals = df.groupby('Team').agg({
        'VORP': 'sum', 'MPG': 'sum', 'USG%': 'sum', 'BPM': 'mean', 'TS%': 'mean'
    }).rename(columns={
        'VORP': 'Team_VORP', 'MPG': 'Team_MPG', 'USG%': 'Team_USG', 'BPM': 'Team_BPM', 'TS%': 'Team_TS'
    })
    df = df.merge(team_totals, on='Team', how='left')

    df['VORP_Share'] = df['VORP'] / df['Team_VORP'].replace(0, 1)
    df['MPG_Share'] = df['MPG'] / df['Team_MPG'].replace(0, 1)
    df['USG_Share'] = df['USG%'] / df['Team_USG'].replace(0, 1)
    df['BPM_Share'] = df['BPM'] / df['Team_BPM'].replace(0, 1)
    df['TS_Norm'] = df['TS%'] / df['Team_TS'].replace(0, 1)

    # Tuned weights from grid search
    alpha, beta, gamma, delta, eta = 0.2, 1.0, 0.8, 0.6, 0.8
    impact = alpha * df['VORP_Share'] + beta * df['BPM_Share']
    role = gamma * df['MPG_Share'] + delta * df['USG_Share']
    df['ARVI'] = impact * role * eta * df['TS_Norm']

    # ========================
    # STEP 3: Team-Level Feature Engineering
    # ========================
    team_stats = df.groupby('Team').agg({
        'ARVI': ['sum', 'mean', 'std', 'max']
    }).reset_index()
    team_stats.columns = ['Team', 'ARVI_Sum', 'ARVI_Mean', 'ARVI_Std', 'ARVI_Max']

    top3_arvi = df.sort_values(['Team', 'ARVI'], ascending=[True, False])\
                  .groupby('Team').head(3).groupby('Team')['ARVI'].sum().reset_index()
    top3_arvi.columns = ['Team', 'Top3_ARVI_Sum']

    team_arvi = team_stats.merge(top3_arvi, on='Team')

    # Count players above team mean ARVI
    team_means = df.groupby('Team')['ARVI'].mean().reset_index().rename(columns={'ARVI': 'Team_ARVI_Mean'})
    df = df.merge(team_means, on='Team', how='left')
    df['Above_Team_Mean'] = df['ARVI'] > df['Team_ARVI_Mean']
    above_mean_counts = df.groupby('Team')['Above_Team_Mean'].sum().reset_index()
    above_mean_counts.columns = ['Team', 'Num_Above_Mean']

    X_df = team_arvi.merge(above_mean_counts, on='Team')

    # ========================
    # STEP 4: Add Actual Wins
    # ========================
    X_df = X_df.merge(team_wins, on='Team')
    X = X_df.drop(columns=['Team', 'Wins'])
    y = X_df['Wins']

    # ========================
    # STEP 5: Train XGBoost Model w/ Grid Search
    # ========================
    param_grid = {
        'n_estimators': [100, 300, 500],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.3],
        'subsample': [0.7, 1.0],
        'colsample_bytree': [0.7, 1.0]
    }

    grid = GridSearchCV(
        estimator=XGBRegressor(random_state=42, objective='reg:squarederror'),
        param_grid=param_grid,
        scoring='neg_root_mean_squared_error',
        cv=5, verbose=1, n_jobs=-1
    )
    grid.fit(X, y)
    best_model = grid.best_estimator_

    # ========================
    # STEP 6: Evaluation & Reporting
    # ========================
    y_pred = best_model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    print("\n✅ Best Hyperparameters:", grid.best_params_)
    print(f"✅ Final RMSE (All Teams): {rmse:.2f}")

    # Save results
    X_df['Predicted_Wins'] = y_pred
    X_df['Error'] = X_df['Predicted_Wins'] - X_df['Wins']
    X_df[['Team', 'Wins', 'Predicted_Wins', 'Error']].to_csv(output_csv, index=False)
    print(f"\n📁 Predictions saved to '{output_csv}'")

    # ========================
    # STEP 7: Visualizations
    # ========================
    plt.figure(figsize=(8,6))
    sns.scatterplot(x='Wins', y='Predicted_Wins', data=X_df, s=70)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
    plt.xlabel("Actual Wins")
    plt.ylabel("Predicted Wins")
    plt.title("XGBoost Predictions vs Actual Wins")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Feature importance
    plt.figure(figsize=(8, 5))
    importances = best_model.feature_importances_
    features = X.columns
    sorted_idx = np.argsort(importances)
    plt.barh(range(len(features)), importances[sorted_idx], align='center')
    plt.yticks(range(len(features)), features[sorted_idx])
    plt.xlabel("Feature Importance")
    plt.title("XGBoost Feature Importances")
    plt.tight_layout()
    plt.show()

    # ========================
    # STEP 8: Correlation Insight
    # ========================
    corr_val, _ = pearsonr(X_df['ARVI_Std'], X_df['Wins'])
    print(f"\n🔍 Correlation between ARVI Std and Wins: {corr_val:.2f}")

    return best_model, X_df