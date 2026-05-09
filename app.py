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
import json
import cohere

load_dotenv()

@st.cache_data
def cached_main2(df, col, file_name):
    return main2(df, col, 5)

@st.cache_data
def cached_main3(df, file_name):
    return main3(df, 3)

if "user_id" not in st.session_state:
    st.session_state.user_id = uuid.uuid4().hex

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)


COHERE_API_KEY=os.getenv("cohere")
co = cohere.Client(COHERE_API_KEY)


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
        st.dataframe(df.head(20))
    else:
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        st.session_state.df = df
        st.session_state.file_name = file.name
        st.dataframe(df.head(20))

    # Load or generate profile report
    if "profile_report" in st.session_state and st.session_state.get("file_name") == file.name:
        html(st.session_state.profile_report.to_html(), height=800, scrolling=True)
    else:
        report = ProfileReport(df, title="Pandas Profiling Report", explorative=True)
        st.session_state.profile_report = report
        html(st.session_state.profile_report.to_html(), height=800, scrolling=True)
    
    if "collection" not in st.session_state or st.session_state.get("file_name") != file.name:
        with st.spinner("Initializing AI analyst..."):
            collection= main4(st.session_state.profile_report, len(df), st.session_state.user_id,co)
            if collection is None:
                st.error("Failed to initialize collection. Check main4() — it must return a ChromaDB collection.")
                st.stop()
            st.session_state.collection = collection
            st.session_state.file_name = file.name  
    
    if "insights" in st.session_state and st.session_state.get("file_name") == file.name:
        st.write(st.session_state.insights)
   
   #Chatbot integration
    with st.sidebar:
      chatbot(st.session_state.collection,co,client)
    
    # Generate insights from context using LLM

    # Structure analysis
    with st.spinner("Analyzing dataset structure..."):
       structure = f"""
       Columns: {list(df.columns)}
       Data Types: {df.dtypes.to_string()}
       """

    # Correlation insights
    def get_corr_insights(df, threshold=0.7):
       numeric_df = df.select_dtypes(include=np.number)
       corr = numeric_df.corr()
       insights = []
       profile=st.session_state.profile_report.to_json()
       profile_dict = json.loads(profile)
       correlations = profile_dict.get("correlations", {}).get("phi_k", {})
       corr_count = 0
       for col1, targets in correlations.items():
          for col2, val in targets.items():
            if col1 < col2 and val > 0.7:  # col1 < col2 ensures we don't duplicate pairs 
                text = f"Strong Relationship: {col1} and {col2} have a correlation of {val:.2f}."
                insights.append(text)
       return "\n".join(insights)
    
    with st.spinner("Analyzing correlations..."):
       correlation_pairs= get_corr_insights(df)

    # Anomalies
    with st.spinner("Detecting anomalies..."):
       numeric_df = df.select_dtypes(include=np.number)
       z_scores = (numeric_df - numeric_df.mean()) / numeric_df.std()
       anomalies = f"""
       z_scores:{z_scores}
       outliers:{numeric_df[(z_scores.abs() > 3).any(axis=1)]}
       row: {numeric_df[(z_scores.abs() > 3).any(axis=1)].index.tolist()}
       """
   
   # missing values , unique values, duplicate rows analysis
    
    def get_missing_info(df):
      missing = df.isnull().sum()
      missing = missing[missing > 0]
      
      return missing
    
    with st.spinner("Analyzing missing data patterns..."):
        missing=get_missing_info(df)
        row=df.shape[0]
        missing_pattern=None
        missing_stats=[]
        if not missing.empty:
           for col, count in missing.items():
              if count > 0.2*row:
                missing_pattern=cached_main2(df,col,st.session_state.file_name) # top 5 features with highest MI with missingness
                missing_stats.append(missing_pattern)
    
    #Cluster analysis
    with st.spinner("Performing cluster analysis..."):
       clusters= cached_main3(df,st.session_state.file_name)
    
    #Domain input
    domain=st.text_input("Enter the domain of your dataset (e.g., finance, healthcare, etc.):")

    if st.button("Generate Insights"):
     with st.spinner("Generating insights..."):
        prompt = f"""
        Act as a data narrator. I have analyzed a dataset and found the following:
        1. STRUCTURE: {structure}
        2. CLUSTERS: {clusters}
        3. CORRELATIONS: {correlation_pairs}
        4. MISSING DATA : {missing_stats}
        5. ANOMALIES: {anomalies}
        TELL THE STORY:
        - Give a name to each cluster based on their behavior.
        - Use the correlations to explain what 'drives' the hero of the story.
        - The Mystery of the Gaps: Analyze the missing values using the lens of MCAR and MAR.
        Identify if the "silence" is a random fluke or a systematic behavior tied to other 
        features.
        Explain the story behind the missingness: are users "hiding" info because of a specific 
        trait, or is there a technical barrier?
        -Based on the anomalies, identify if there are "plot twists" in the data that could indicate hidden
        subgroups or rare events and if data is reliable.
        
        FORMAT:
      - One-sentence 'Headline'.
      - 'The Plot' (The main trends).
      - 'The Mystery' (Analysis of missing info).
      - 'The Recommendation' (Next steps). 
      - Keep it concise 2-3 points for each section, making it engaging, like a news article summarizing the key insights from the dataset.
      """
        
        response = client.chat.completions.create(model="openai/gpt-oss-120b", messages=[{"role": "user", "content": prompt}],temperature=0.2)
        st.subheader("Insights:")
        st.write(response.choices[0].message.content)
        st.session_state.insights = response.choices[0].message.content
     

     
