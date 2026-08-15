"""
Shared preprocessing logic for the Ames Housing price model.
Used both when training/saving artifacts (train_and_save.py) and
when serving predictions (main.py), so the two never drift apart.
"""

import pandas as pd
import numpy as np

COLS_NONE = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
             'GarageQual', 'GarageFinish', 'GarageType', 'GarageCond',
             'BsmtFinType2', 'BsmtExposure', 'BsmtCond', 'BsmtQual',
             'BsmtFinType1', 'MasVnrType']

QUAL_MAP = {'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5, 'None': 0}
QUAL_COLS = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC',
             'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 'PoolQC']

LOTSHAPE_MAP = {'Reg': 4, 'IR1': 3, 'IR2': 2, 'IR3': 1}
UTILITIES_MAP = {'AllPub': 4, 'NoSewr': 3, 'NoSeWa': 2, 'ELO': 1}
LANDSLOPE_MAP = {'Gtl': 3, 'Mod': 2, 'Sev': 1}
BSMTEXPOSURE_MAP = {'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'None': 0}
BSMTFIN_MAP = {'GLQ': 6, 'ALQ': 5, 'BLQ': 4, 'Rec': 3, 'LwQ': 2, 'Unf': 1, 'None': 0}
FUNCTIONAL_MAP = {'Typ': 8, 'Min1': 7, 'Min2': 6, 'Mod': 5, 'Maj1': 4, 'Maj2': 3, 'Sev': 2, 'Sal': 1}
GARAGEFINISH_MAP = {'Fin': 3, 'RFn': 2, 'Unf': 1, 'None': 0}
PAVEDDRIVE_MAP = {'Y': 2, 'P': 1, 'N': 0}
FENCE_MAP = {'GdPrv': 4, 'MnPrv': 3, 'GdWo': 2, 'MnWw': 1, 'None': 0}

NOMINAL_COLS = ['MSZoning', 'Street', 'Alley', 'LandContour', 'LotConfig',
                'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
                'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
                'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating',
                'CentralAir', 'Electrical', 'GarageType', 'MiscFeature',
                'SaleType', 'SaleCondition']


def clean_and_encode(df: pd.DataFrame) -> pd.DataFrame:
    """Applies the fill + ordinal-encoding steps to a raw dataframe.
    One-hot encoding is done separately (via pd.get_dummies) since it
    needs to happen jointly across train/test/inference rows to keep
    columns consistent — see notes.md for why."""
    df = df.copy()

    for col in COLS_NONE:
        if col in df.columns:
            df[col] = df[col].fillna('None')

    if 'GarageYrBlt' in df.columns:
        df['GarageYrBlt'] = df['GarageYrBlt'].fillna(0)
    if 'MasVnrArea' in df.columns:
        df['MasVnrArea'] = df['MasVnrArea'].fillna(0)

    for col in QUAL_COLS:
        if col in df.columns:
            df[col] = df[col].map(QUAL_MAP)

    if 'LotShape' in df.columns:
        df['LotShape'] = df['LotShape'].map(LOTSHAPE_MAP)
    if 'Utilities' in df.columns:
        df['Utilities'] = df['Utilities'].map(UTILITIES_MAP)
    if 'LandSlope' in df.columns:
        df['LandSlope'] = df['LandSlope'].map(LANDSLOPE_MAP)
    if 'BsmtExposure' in df.columns:
        df['BsmtExposure'] = df['BsmtExposure'].map(BSMTEXPOSURE_MAP)
    if 'BsmtFinType1' in df.columns:
        df['BsmtFinType1'] = df['BsmtFinType1'].map(BSMTFIN_MAP)
    if 'BsmtFinType2' in df.columns:
        df['BsmtFinType2'] = df['BsmtFinType2'].map(BSMTFIN_MAP)
    if 'Functional' in df.columns:
        df['Functional'] = df['Functional'].map(FUNCTIONAL_MAP)
    if 'GarageFinish' in df.columns:
        df['GarageFinish'] = df['GarageFinish'].map(GARAGEFINISH_MAP)
    if 'PavedDrive' in df.columns:
        df['PavedDrive'] = df['PavedDrive'].map(PAVEDDRIVE_MAP)
    if 'Fence' in df.columns:
        df['Fence'] = df['Fence'].map(FENCE_MAP)

    return df


def align_to_training_columns(df: pd.DataFrame, feature_columns: list) -> pd.DataFrame:
    """One-hot encodes nominal columns, then reindexes to the exact
    column set the model was trained on (fills anything missing with 0).
    This is what keeps a single inference row consistent with training,
    the same way we combined train+test before encoding during training."""
    present_nominal = [c for c in NOMINAL_COLS if c in df.columns]
    df = pd.get_dummies(df, columns=present_nominal, drop_first=True)
    df = df.reindex(columns=feature_columns, fill_value=0)
    return df
