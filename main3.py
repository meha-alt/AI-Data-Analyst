from sklearn.cluster import KMeans
from kmodes.kprototypes import KPrototypes
from kmodes.kmodes import KModes
import pandas as pd
import numpy as np

def main3(df, k):
    df1 = df.copy()
    num_cols = df1.select_dtypes(include=[np.number]).columns
    cat_cols = df1.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        df1[col] = df1[col].astype(str)
    if len(cat_cols) == 0 and len(num_cols) > 0:
        df1[num_cols] = df1[num_cols].fillna(df1[num_cols].median())
        model = KMeans(n_clusters=k, random_state=42)
        df1["Cluster"] = model.fit_predict(df1[num_cols])
    elif len(num_cols) == 0 and len(cat_cols) > 0:
        df1[cat_cols] = df1[cat_cols].fillna("missing")
        model = KModes(n_clusters=k, init="Cao", verbose=0)
        df1["Cluster"] = model.fit_predict(df1[cat_cols])
    else:
        for col in num_cols:
            df1[col] = df1[col].fillna(df1[col].median())

        for col in cat_cols:
            df1[col] = df1[col].fillna(df1[col].mode()[0])

        cat_idx = [df1.columns.get_loc(c) for c in cat_cols]

        model = KPrototypes(n_clusters=k, init="Cao", verbose=0)
        df1["Cluster"] = model.fit_predict(df1.values, categorical=cat_idx)

    cluster_summary = []

    for cluster_id in sorted(df1["Cluster"].unique()):
        cluster_data = df1[df1["Cluster"] == cluster_id]

        summary = {
            "Cluster_ID": f"Group_{cluster_id}",
            "Size_Percent": round(len(cluster_data) / len(df1) * 100, 2),
            "Key_Characteristics": cluster_data.mean(numeric_only=True).to_dict(),
            "Variance": cluster_data.var(numeric_only=True).mean(),
            "Top_Categories": {
                col: cluster_data[col].mode()[0]
                for col in cat_cols
            }
        }

        cluster_summary.append(summary)
    
    for c in cluster_summary:
        c["Score"] = c["Size_Percent"]

    top_5_clusters = sorted(cluster_summary, key=lambda x: x["Score"], reverse=True)[:5]

    return str(top_5_clusters)