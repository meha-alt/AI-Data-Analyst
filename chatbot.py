import streamlit as st
from groq import Groq
import os
from dotenv import load_dotenv



load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

def chatbot(collection):
    st.header("Chat with AI Analyst")

    if "collection" not in st.session_state:
        st.session_state.collection = collection

    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
           st.write(msg["content"])
   
    query = st.chat_input(
        "Ask your question about the dataset:"
       )
    
    st.caption("""
    Ask about:
    • data quality issues  
    • missing values  
    • correlations  
    • preprocessing suggestions  
    • feature engineering ideas  
    • skewness and outliers  
    • model readiness""")
    
    if query:
        with st.chat_message("user"):
            st.write(query)
       
        st.session_state.messages.append({
            "role": "user",
            "content": query
        })
          
        results = st.session_state.collection.query(
            query_texts=[query],
            n_results=3 )
          
        context = "\n".join(
            results["documents"][0])
          
        messages = [
            {
                "role": "system",
                "content": f"""
                You are an expert AI data analyst.

                Use these dataset insights:
                {context}

                Answer questions clearly and professionally.
                Answer should be brief and clear in 4-5 lines.
                """
            }]

        messages.extend(
            st.session_state.messages
          )

        
        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages,
            temperature=0.2
         )

        answer = response.choices[0].message.content
        
        with st.chat_message("assistant"):
            st.write(answer)
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
          })
          
        
          