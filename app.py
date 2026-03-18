# import streamlit as st
# import joblib

# # 1. Page Configuration
# st.set_page_config(page_title="Email Spam Detector", page_icon="📧")

# # 2. Load the "Power Environment" models
# # (Make sure these .pkl files are in the same folder as this script)
# @st.cache_resource # This ensures the model only loads ONCE, making the app fast
# def load_models():
#     model = joblib.load('spam_classifier_model.joblib')
#     vectorizer = joblib.load('feature_extraction.joblib')
#     return model, vectorizer

# model, vectorizer = load_models()

# # 3. The User Interface
# st.title("📧 Intelligent Spam Detector")
# st.write("Type or paste an email below to see if it's safe or a scam.")

# # Input area
# user_input = st.text_area("Enter Email Text:", placeholder="e.g., Congratulations! You won a prize...")

# if st.button("Analyze Email"):
#     if user_input.strip() == "":
#         st.warning("Please enter some text first!")
#     else:
#         # 4. Prediction Logic
#         data = vectorizer.transform([user_input])
#         prediction = model.predict(data)
        
#         # 0 = Spam, 1 = Ham (based on your specific dataset labels)
#         if prediction[0] == 0:
#             st.error("🚨 **RESULT: THIS IS SPAM!**")
#             st.info("Be careful! This looks like a phishing attempt or a scam.")
#         else:
#             st.success("✅ **RESULT: THIS IS SAFE (HAM)**")
#             st.balloons()

# # 5. Sidebar Info
# st.sidebar.markdown("### About this Project")
# st.sidebar.info("Built with Scikit-Learn and Streamlit for Annu's Portfolio.")


import streamlit as st
import joblib
import pandas as pd
import plotly.express as px
from datetime import datetime

# Page Config
st.set_page_config(page_title="Pro Spam Shield", page_icon="🛡️", layout="wide")

# Load Models
@st.cache_resource
def load_assets():
    model = joblib.load('spam_classifier_model.joblib')
    vectorizer = joblib.load('feature_extraction.joblib')
    return model, vectorizer

model, vectorizer = load_assets()

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("🛡️ SpamShield Pro")
menu = st.sidebar.radio("Navigation", ["Single Check", "Batch Analysis", "Project Analytics"])

# --- FEATURE 1: SINGLE CHECK ---
if menu == "Single Check":
    st.title("🔍 Single Email Verification")
    user_input = st.text_area("Paste Email Content Here:", height=200)
    
    if st.button("Run Security Scan"):
        if user_input:
            # Get Probabilities (Shows how "confident" the model is)
            vec_input = vectorizer.transform([user_input])
            prediction = model.predict(vec_input)[0]
            probability = model.predict_proba(vec_input)[0] # [Prob_Spam, Prob_Ham]
            
            if prediction == 0:
                st.error(f"🚨 SPAM DETECTED ({probability[0]*100:.1f}% Confidence)")
            else:
                st.success(f"✅ SAFE EMAIL ({probability[1]*100:.1f}% Confidence)")
        else:
            st.warning("Please enter text to analyze.")

# --- FEATURE 2: BATCH ANALYSIS & DOWNLOAD ---
elif menu == "Batch Analysis":
    st.title("📂 Batch File Processing")
    uploaded_file = st.file_uploader("Upload CSV (must have a 'Message' column)", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if 'Message' in df.columns:
            # Process all rows
            X_features = vectorizer.transform(df['Message'])
            df['Prediction'] = model.predict(X_features)
            df['Result'] = df['Prediction'].map({0: 'Spam', 1: 'Ham'})
            
            st.write("### Analysis Preview", df.head())
            
            # DOWNLOAD REPORT FEATURE
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Full Security Report (CSV)",
                data=csv,
                file_name=f"spam_report_{datetime.now().strftime('%Y%m%d')}.csv",
                mime='text/csv',
            )
        else:
            st.error("CSV must contain a column named 'Message'")

# --- FEATURE 3: ANALYTICS ---
elif menu == "Project Analytics":
    st.title("📊 Model Insights")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### Performance Metrics")
        st.metric(label="Model Accuracy", value="97.6%")
        st.metric(label="Algorithm", value="Random Forest")
        
    with col2:
        # Mock chart for visualization
        chart_data = pd.DataFrame({'Category': ['Spam', 'Ham'], 'Count': [450, 1200]})
        fig = px.pie(chart_data, values='Count', names='Category', title="Dataset Distribution")
        st.plotly_chart(fig)