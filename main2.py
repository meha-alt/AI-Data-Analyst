import pandas as pd
from sklearn.feature_selection import mutual_info_classif

def main2(df, col, top_n):

    temp = df.copy()
    temp["missing_flag"] = temp[col].isnull().astype(int)
    X = temp.drop(columns=[col])
    X = pd.get_dummies(X, drop_first=True)
    X = X.fillna(0)
    mi = mutual_info_classif(X, temp["missing_flag"], random_state=42)
    mi_series = pd.Series(mi, index=X.columns)
    mi_series = mi_series.sort_values(ascending=False)
    return {
        "column": col,
        "missing_ratio": df[col].isnull().mean(),
        "top_mi_features": mi_series.head(top_n)
    }