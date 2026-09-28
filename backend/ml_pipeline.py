import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
import lightgbm as lgb
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_artifacts.joblib")

def preprocess_data(df, is_train=True, training_columns=None):
    df_copy = df.copy()
    
    # 1. Fill NaNs
    none_cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
                 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
                 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                 'MasVnrType']
    for col in none_cols:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].fillna('None')

    zero_cols = ['GarageYrBlt', 'GarageArea', 'GarageCars',
                 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
                 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrArea']
    for col in zero_cols:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].fillna(0)

    if 'Neighborhood' in df_copy.columns and 'LotFrontage' in df_copy.columns:
        if is_train:
            df_copy['LotFrontage'] = df_copy.groupby('Neighborhood')['LotFrontage'].transform(lambda x: x.fillna(x.median()))
        else:
            # During inference, single row might not have group median. Fallback to global median (70 is approx global median).
            df_copy['LotFrontage'] = df_copy['LotFrontage'].fillna(70.0)
            
    if 'GarageYrBlt' in df_copy.columns and 'YearBuilt' in df_copy.columns:
        df_copy['GarageYrBlt'] = np.where(df_copy['GarageYrBlt'] == 0, df_copy['YearBuilt'], df_copy['GarageYrBlt'])

    # Fill remaining with mode/median
    for col in df_copy.columns:
        if df_copy[col].isnull().sum() > 0:
            if pd.api.types.is_numeric_dtype(df_copy[col]):
                df_copy[col] = df_copy[col].fillna(0) # Simplify for production
            else:
                df_copy[col] = df_copy[col].fillna(df_copy[col].mode()[0] if not df_copy[col].mode().empty else 'None')

    # 2. Feature Engineering
    if 'YrSold' in df_copy.columns and 'YearBuilt' in df_copy.columns:
        df_copy['AgeAtSale'] = (df_copy['YrSold'] - df_copy['YearBuilt']).clip(lower=0)
    if 'YrSold' in df_copy.columns and 'YearRemodAdd' in df_copy.columns:
        df_copy['AgeSinceRemod'] = (df_copy['YrSold'] - df_copy['YearRemodAdd']).clip(lower=0)
    
    if all(c in df_copy.columns for c in ['TotalBsmtSF', '1stFlrSF', '2ndFlrSF']):
        df_copy['TotalSF'] = df_copy['TotalBsmtSF'] + df_copy['1stFlrSF'] + df_copy['2ndFlrSF']
    if all(c in df_copy.columns for c in ['OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch']):
        df_copy['TotalPorchSF'] = df_copy['OpenPorchSF'] + df_copy['EnclosedPorch'] + df_copy['3SsnPorch'] + df_copy['ScreenPorch']
    if all(c in df_copy.columns for c in ['FullBath', 'HalfBath', 'BsmtFullBath', 'BsmtHalfBath']):
        df_copy['TotalBathrooms'] = df_copy['FullBath'] + 0.5 * df_copy['HalfBath'] + df_copy['BsmtFullBath'] + 0.5 * df_copy['BsmtHalfBath']

    if 'PoolArea' in df_copy.columns: df_copy['HasPool'] = (df_copy['PoolArea'] > 0).astype(int)
    if 'GarageArea' in df_copy.columns: df_copy['HasGarage'] = (df_copy['GarageArea'] > 0).astype(int)
    if 'TotalBsmtSF' in df_copy.columns: df_copy['HasBsmt'] = (df_copy['TotalBsmtSF'] > 0).astype(int)
    if 'Fireplaces' in df_copy.columns: df_copy['HasFireplace'] = (df_copy['Fireplaces'] > 0).astype(int)
    if 'YearRemodAdd' in df_copy.columns and 'YearBuilt' in df_copy.columns:
        df_copy['IsRemodeled'] = (df_copy['YearRemodAdd'] != df_copy['YearBuilt']).astype(int)

    # 3. Encoding
    if 'MSSubClass' in df_copy.columns: df_copy['MSSubClass'] = df_copy['MSSubClass'].astype(str)
    if 'YrSold' in df_copy.columns: df_copy['YrSold'] = df_copy['YrSold'].astype(str)
    if 'MoSold' in df_copy.columns: df_copy['MoSold'] = df_copy['MoSold'].astype(str)

    qual_mapping = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'None': 0}
    ordinal_cols = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC',
                    'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']
    for col in ordinal_cols:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].map(qual_mapping).fillna(0)

    if 'BsmtExposure' in df_copy.columns:
        df_copy['BsmtExposure'] = df_copy['BsmtExposure'].map({'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'None': 0}).fillna(0)
    if 'Functional' in df_copy.columns:
        df_copy['Functional'] = df_copy['Functional'].map({'Typ': 7, 'Min1': 6, 'Min2': 5, 'Mod': 4, 'Maj1': 3, 'Maj2': 2, 'Sev': 1, 'Sal': 0}).fillna(7)

    if 'Id' in df_copy.columns:
        df_copy = df_copy.drop(columns=['Id'])
    
    # 4. One-Hot Encoding
    df_encoded = pd.get_dummies(df_copy, drop_first=False)
    
    if is_train:
        return df_encoded
    else:
        # Align columns
        for col in training_columns:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        df_encoded = df_encoded[training_columns]
        # Fill any remaining NaNs after reindexing
        df_encoded = df_encoded.fillna(0)
        return df_encoded


def train_and_export(train_csv_path):
    train = pd.read_csv(train_csv_path)
    
    # Remove outliers
    outliers_idx = train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 300000)].index
    train = train.drop(index=outliers_idx).reset_index(drop=True)
    
    y_train = train["SalePrice"]
    y_train_log = np.log1p(y_train)
    
    X_train_raw = train.drop(columns=["SalePrice"])
    X_train = preprocess_data(X_train_raw, is_train=True)
    training_columns = X_train.columns.tolist()
    
    lasso_selector = SelectFromModel(Lasso(alpha=0.0005, max_iter=10000, random_state=42))

    eval_pipelines = {
        'Ridge': Pipeline([('scaler', RobustScaler()), ('selector', lasso_selector), ('model', Ridge(alpha=15.0))]),
        'Lasso': Pipeline([('scaler', RobustScaler()), ('model', Lasso(alpha=0.0004, max_iter=10000, random_state=42))]),
        'GBR': Pipeline([('scaler', RobustScaler()), ('selector', lasso_selector), 
                         ('model', GradientBoostingRegressor(n_estimators=450, learning_rate=0.03, max_depth=3, max_features='sqrt', min_samples_leaf=15, loss='huber', random_state=42))]),
        'LGBM': Pipeline([('scaler', RobustScaler()), ('selector', lasso_selector),
                          ('model', lgb.LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=20, max_depth=4, subsample=0.8, colsample_bytree=0.7, min_child_samples=15, reg_alpha=0.1, reg_lambda=0.5, random_state=42, verbose=-1))])
    }
    
    print("Training models...")
    for name, pipe in eval_pipelines.items():
        pipe.fit(X_train, y_train_log)
        
    # Simplified optimization of weights (Normally done via OOF, doing equal weights for simplicity in production if fast, or we can use fixed robust weights)
    # Using fixed weights that usually perform well on Ames
    opt_weights = np.array([0.15, 0.15, 0.35, 0.35])
    
    artifacts = {
        'models': eval_pipelines,
        'weights': opt_weights,
        'columns': training_columns
    }
    
    joblib.dump(artifacts, MODEL_PATH)
    print(f"Model exported to {MODEL_PATH}")

if __name__ == '__main__':
    # Train using parent directory train.csv
    train_csv_path = os.path.join(os.path.dirname(__file__), "..", "..", "train.csv")
    train_and_export(train_csv_path)
