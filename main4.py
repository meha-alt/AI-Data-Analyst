from ydata_profiling import ProfileReport
import chromadb
import json


def main4(profile,l,user_id,cohere_client):
    client = chromadb.Client()
    collection_name = f"profile_{user_id}"  # unique per user session
    try:
      client.delete_collection(name=collection_name) # Clear previous data for this session
    
    except:
      pass
    # create fresh collection
    collection = client.get_or_create_collection(name=collection_name)
    
    profile=profile.to_json()
    profile_dict = json.loads(profile)
   
    documents = []
    metadatas = []
    ids = []
    
    # GLOBAL SUMMARY
    table_stats = profile_dict.get("table", {})
    global_text = (
        f"Dataset Overview: {table_stats.get('n_var')} variables, "
        f"{table_stats.get('n_obs')} observations. "
        f"Total missing cells: {table_stats.get('n_cells_missing')} "
        f"({table_stats.get('p_cells_missing'):.2%})."
    )
    documents.append(global_text)
    metadatas.append({"category": "global_summary", "column": "all"})
    ids.append("global_001")

    # VARIABLE INSIGHTS 
    variables = profile_dict.get("variables", {})
    for col, details in variables.items():
        v_type = details.get('type')
        text = f"Variable '{col}' ({v_type}). "
        n_unique = details.get('n_unique', 0)
        p_unique = details.get('p_unique', 0) # Percentage unique
        text += f"Has {n_unique} unique values ({p_unique:.1%}). "
        # Main focus on skewness and entropy
        if v_type == "Numeric":
            skew = details.get('skewness', 0)
            text += f"Mean: {details.get('mean'):.2f}, Skewness: {skew:.2f}. "
            if abs(skew) > 1:
                text += "Critical: This column is highly skewed and may need transformation. "
        
        if details.get('p_missing', 0) > 0:
            text += f"Warning: {details.get('p_missing'):.1%} missing data."
        
        p_duplicates = details.get('p_duplicates', 0)
        if p_duplicates > 0.5: # If more than 50% is repeated
            text += f"High Redundancy: {p_duplicates:.1%} are duplicate entries. "
        
        documents.append(text)
        metadatas.append({"category": "variable_detail", "column": col})
        ids.append(f"var_{col}")

    # ALERTS (data quality issues)
    alerts = profile_dict.get("alerts", [])
    for i, alert in enumerate(alerts):
        documents.append(f"Data Quality Alert: {alert}")
        metadatas.append({"category": "alert", "column": "multiple"})
        ids.append(f"alert_{i}")

    # PHIK CORRELATIONS 
    # Highlighting non-linear relationships for both numeric and categorical variables using phi_k correlation.
    # We only include strong correlations ( > 0.7) to keep it concise.
    correlations = profile_dict.get("correlations", {}).get("phi_k", {})
    corr_count = 0
    for col1, targets in correlations.items():
        for col2, val in targets.items():
            if col1 < col2 and val > 0.7:  # col1 < col2 ensures we don't duplicate pairs 
                text = f"Strong Relationship: {col1} and {col2} have a correlation of {val:.2f}."
                documents.append(text)
                metadatas.append({"category": "correlation", "column": f"{col1}_{col2}"})
                ids.append(f"corr_{corr_count}")
                corr_count += 1
    
    correlations2 = profile_dict.get("correlations", {}).get("pearson", {})
    for col1, targets in correlations2.items():
        for col2, val in targets.items():
            if col1 < col2 and abs(val) > 0.7:  # col1 < col2 ensures we don't duplicate pairs 
                text = f"Strong Linear Relationship: {col1} and {col2} have a Pearson correlation of {val:.2f}."
                documents.append(text)
                metadatas.append({"category": "correlation", "column": f"{col1}_{col2}"})
                ids.append(f"corr_pearson_{corr_count}")
                corr_count += 1
    
    correlations3 = profile_dict.get("correlations", {}).get("spearman", {})
    for col1, targets in correlations3.items():
        for col2, val in targets.items():
            if col1 < col2 and abs(val) > 0.7:  # col1 < col2 ensures we don't duplicate pairs 
                text = f"Strong Monotonic Relationship: {col1} and {col2} have a Spearman correlation of {val:.2f}."
                metadatas.append({"category": "correlation", "column": f"{col1}_{col2}"})
                ids.append(f"corr_spearman_{corr_count}")
                corr_count += 1

    # Batch Embed and Add to Chroma
    response = cohere_client.embed(
    texts=documents,
    model="embed-english-v3.0",
    input_type="search_document")
    
    embeddings = response.embeddings
    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )
    
    return collection
