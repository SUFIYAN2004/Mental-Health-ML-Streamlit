import streamlit as st
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd
import joblib


st.header("Teenager's mental health predictor...")
st.write("------------------------------------------------")


@st.cache_resource
def Load_model():
    gender_le = joblib.load("gender.joblib")
    platform_le = joblib.load("platform.joblib")
    Social_le = joblib.load("social.joblib")
    model = joblib.load("teenager_mental_health_model.joblib")
    return gender_le, platform_le, Social_le, model

gender_le, platform_le, Social_le, model = Load_model()


with st.sidebar:

    age = st.slider("Select your age: ", min_value=13, max_value=20)
    gender = st.selectbox("Select your gender: ", ('male', 'female'))
    daily_social_media_hours = st.slider("Select your hours you spend in Social media: ", min_value=0.0, max_value=8.0)
    platform_usage = st.selectbox("Select platform: ", ('Instagram', 'TikTok', 'Both'))
    sleep_hours = st.slider("Select your Sleep Hours: ", min_value=1.0, max_value=8.0)
    screen_time_before_sleep = st.slider("After the screen sleep time: ", min_value=0.5, max_value=3.0)
    academic_performance = st.slider("Study Performance: ", min_value=2.0, max_value=4.0)
    physical_activity = st.slider("physical Activity: ", min_value=0.0, max_value=2.0)
    social_interaction_level = st.selectbox("Social Life Interaction: ", ("medium", "low", "high"))
    stress_level = st.slider("Select your Stress level: ", min_value=1, max_value=10)
    anxiety_level = st.slider("Select your anxiety_level: ", min_value=1, max_value=10)
    addiction_level = st.slider("Select your Addition level: ", min_value=1, max_value=10)

    button = st.button("predict", type="primary")

if button:
    genders = gender_le.transform([gender])[0]
    platform_usages = platform_le.transform([platform_usage])[0]
    social_interaction_levels =Social_le.transform([social_interaction_level])[0]
    test = [age, genders, daily_social_media_hours, platform_usages, sleep_hours, screen_time_before_sleep, academic_performance, physical_activity, social_interaction_levels, stress_level, anxiety_level, addiction_level]
    prediction = model.predict([test])[0]
    probabilites = model.predict_proba([test])
    risk_probs = probabilites[0][1]
    risk_percentage  =  risk_probs * 100
    if prediction == 0:
        st.header("You are Free From Depression!")
        st.success(f"The model is {100 - risk_percentage:.1f}% confident in this result.")
    else:
        st.header("Take care of your health, you have Depression")
        st.error(f"The model is {risk_percentage:.1f}% confident in this result. Please take care!")


    st.markdown("---")
    st.header("How did the AI make this decision?")
    st.write("Bars pointing **UP** push towards depression. Bars pointing **DOWN** protect against it.")


    feature_names = [
        "Age", "Gender", "Social Media Hours", "Platform", 
        "Sleep Hours", "Screen Time in Bed", "Study Performance", 
        "Physical Activity", "Social Interaction", "Stress Level", 
        "Anxiety Level", "Addiction Level"
    ]


    coefficients = model.coef_[0]


    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Impact": coefficients
    })


    importance_df = importance_df.sort_values(by="Impact", ascending=True)


    st.bar_chart(data=importance_df, x="Feature", y="Impact")


st.markdown("-------------")
st.caption(":red[**Disclaimer:** This model can make mistakes, please check with a medical specialist or professional counselor.]")