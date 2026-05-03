import streamlit as st
import pandas as pd
import numpy as np
from ydata_profiling import ProfileReport
from streamlit.components.v1 import html
import os
from io import BytesIO
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

file=st.file_uploader("Upload your Excel or CSV file", type=[".xls", ".xlsx", ".csv"])

if file is not None:
    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    st.write("Dataframe:")
    st.dataframe(df)

    profile = ProfileReport(df, title="Pandas Profiling Report", explorative=True)
    html(profile.to_html(), height=800, scrolling=True)

    # Generate insights using Groq
    context_1 = f"""
    Columns: {list(df.columns)}
    Data Types: {df.dtypes.to_string()}
    Sample Data:
    {df.head(5).to_string()}
    """

    context_2 = f"""
    Summary Statistics:
    {df.describe(include='all').to_string()}
    """
    def get_corr_insights(df, threshold=0.5):
       corr = df.corr(numeric_only=True)
       insights = []
    
       for i in range(len(corr.columns)):
          for j in range(i+1, len(corr.columns)):
             val = corr.iloc[i, j]
             if abs(val) > threshold:
                relation = "positively" if val > 0 else "negatively"
                insights.append(
                    f"{corr.columns[i]} is {relation} correlated with {corr.columns[j]} ({val:.2f})"
                )
       return "\n".join(insights)

    context_3 = get_corr_insights(df)

    #anamolies
    numeric_df = df.select_dtypes(include=np.number)
    z_scores = (numeric_df - numeric_df.mean()) / numeric_df.std()
    context_4 = f"""
    z_scores:{z_scores}
    outliers:{numeric_df[(z_scores.abs() > 3).any(axis=1)]}
    """

    def get_missing_info(df):
      missing = df.isnull().sum()
      missing = missing[missing > 0]
      if missing.empty:
        return "No missing values found."
      return "\n".join([
        f"{col}: {count} missing values"
        for col, count in missing.items()])
    
    def get_unique_info(df):
       info = []
       for col in df.columns:
         unique_count = df[col].nunique()
         info.append(f"{col}: {unique_count} unique values")
       return "\n".join(info)
    
    def get_duplicate_info(df):
       duplicate_rows = df[df.duplicated()]
       count = len(duplicate_rows)
       if count == 0:
        return "No duplicate rows found.", None
       return f"{count} duplicate rows found.", duplicate_rows
    
    
    domain=st.text_input("Enter the domain of your dataset (e.g., finance, healthcare, etc.):")

    if st.button("Generate Insights"):
     prompt = f"""
     You are a data analyst. Based on the following context, provide insights about the dataset in the domain of {domain}.

     STRUCTURE:
     {context_1}

     SUMMARY STATISTICS:
     {context_2}

     KEY RELATIONSHIPS:
     {context_3}

     ANALOMALIES:
    {context_4}These are unusual data points—explain why they might exist and their potential impact.

    DATA QUALITY REPORT:
    Missing Values:
    {get_missing_info(df)}

    Duplicate Rows:
    {get_duplicate_info(df)}

    Unique Values:
    {get_unique_info(df)}
     
     Your task:
     1. Identify the 2-3 MOST IMPORTANT insights (not generic)
     2. Highlight any strong correlations and explain WHY they might exist
     3. Detect anomalies or unusual patterns
     4. Provide actionable recommendations (what should be done next)

     Rules:
     - Be specific, avoid generic statements
     - Focus on decision-making value
     - Use simple language but deep reasoning
     - Highlight numbers clearly
     - Keep each point under 2 lines
     - Focus on decision-making impact.- Add "impact" after each insight.

     Output format:
     ### Key Insights
     - ...

     ### Correlations Explained
     - ...

    ### Risks / Anomalies
    - ...

    ### Recommendations
    - ...
    """
     response = client.chat.completions.create(model="openai/gpt-oss-120b", messages=[{"role": "user", "content": prompt}])
     st.subheader("Insights:")
     st.write(response.choices[0].message.content)

