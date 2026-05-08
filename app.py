import streamlit as st
import pandas as pd
import numpy as np
from ydata_profiling import ProfileReport
from streamlit.components.v1 import html
import os
from io import BytesIO
from dotenv import load_dotenv
from groq import Groq
from main2 import main2
from main3 import main3
from main4 import main4
from chatbot import chatbot
import uuid

load_dotenv()

if "user_id" not in st.session_state:
    st.session_state.user_id = uuid.uuid4().hex

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

st.markdown("<h1 style='text-align: center;'>AI Data Analyst</h1>", unsafe_allow_html=True)

file=st.file_uploader("Upload your Excel or CSV file", type=[".xls", ".xlsx", ".csv"])

if file is None or file.name != st.session_state.get("file_name"):
   current_user_id = st.session_state.get("user_id")
   st.session_state.clear()
   st.session_state.user_id = current_user_id
   st.session_state.file_name = file.name if file is not None else None


if file is not None:
    if "df" in st.session_state and st.session_state.get("file_name") == file.name:
        df = st.session_state.df
        st.dataframe(df)
    else:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        st.session_state.df = df
        st.session_state.file_name = file.name
        st.dataframe(df)

    # Load or generate profile report
    if "profile_report" in st.session_state and st.session_state.get("file_name") == file.name:
        html(st.session_state.profile_report.to_html(), height=800, scrolling=True)
    else:
        report = ProfileReport(df, title="Pandas Profiling Report", explorative=True)
        st.session_state.profile_report = report
        html(st.session_state.profile_report.to_html(), height=800, scrolling=True)

    
    if "collection" not in st.session_state or st.session_state.get("file_name") != file.name:
        with st.spinner("Initializing AI analyst..."):
            collection = main4(st.session_state.profile_report, len(df), st.session_state.user_id)
            if collection is None:
                st.error("Failed to initialize collection. Check main4() — it must return a ChromaDB collection.")
                st.stop()
            st.session_state.collection = collection
            st.session_state.file_name = file.name  
    
    if "insights" in st.session_state and st.session_state.get("file_name") == file.name:
        st.write(st.session_state.insights)
   
   #Chatbot integration
    with st.sidebar:
      chatbot(st.session_state.collection)
    
    # Generate insights from context using LLM

    # Structure analysis
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
    # Correlation insights
    
    def get_corr_insights(df, threshold=0.7):
       numeric_df = df.select_dtypes(include=np.number)
       corr = numeric_df.corr()
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

    # Anomalies
    
    numeric_df = df.select_dtypes(include=np.number)
    z_scores = (numeric_df - numeric_df.mean()) / numeric_df.std()
    context_4 = f"""
    z_scores:{z_scores}
    outliers:{numeric_df[(z_scores.abs() > 3).any(axis=1)]}
    row: {numeric_df[(z_scores.abs() > 3).any(axis=1)].index.tolist()}
    """
   
   # missing values , unique values, duplicate rows analysis
    
    def get_missing_info(df):
      missing = df.isnull().sum()
      missing = missing[missing > 0]
      
      return missing
    
    missing=get_missing_info(df)
    row=df.shape[0]
    missing_pattern=None
    if not missing.empty:
        for col, count in missing.items():
            if count > 0.2*row:
                missing_pattern=main2(df,col,5) # top 5 features with highest MI with missingness

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
    
    #Cluster analysis
    context_5=main3(df,3)
    
    #Domain input
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

      CLUSTER ANALYSIS:
    {context_5}Interpret the given cluster summaries in simple business terms.
    For each cluster:
       - Describe the group in 1–2 lines
       - List key patterns (bullets)
       - Mention what makes it different
       - Give a real-world meaning (user segment)
       - Suggest one actionable insight per cluster
       - Do NOT repeat raw numbers
       - Focus on patterns and meaning only
       - Be concise and clear

    DATA QUALITY REPORT:
    Missing Values:
    {missing_pattern if missing_pattern else missing} - Explain relationships between missingness and other features.
    Classify missingness (MCAR, MAR, MNAR) with brief reasoning.Suggest appropriate handling strategies.


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
     -Interpret patterns using non-causal language (e.g., ‘suggests’, ‘may indicate’).

     Output format:
     ### Key Insights
     - ...

     ### Correlations Explained
     - ...

    ### Risks / Anomalies
    - ...
    
    ### Cluster Interpretations
    - ...
    
    ### Data Quality Issues
    - ...

    ### Recommendations
    - ...


    """
     response = client.chat.completions.create(model="openai/gpt-oss-120b", messages=[{"role": "user", "content": prompt}],temperature=0.2)
     st.subheader("Insights:")
     st.write(response.choices[0].message.content)
     st.session_state.insights = response.choices[0].message.content
     

     
