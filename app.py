import streamlit as st
import numpy as np
import pickle

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

st.set_page_config(page_title="Course Purchase Predictor", page_icon="🎓")

st.title("🎓 EdTech Course Purchase Predictor")
st.markdown("Enter student details below to predict if they will purchase a course.")
st.divider()

# Input fields
age = st.slider("Age", 15, 60, 22)
study_hours = st.slider("Study Hours Per Week", 0, 40, 14)
courses_completed = st.slider("Previous Courses Completed", 0, 20, 3)
platform_visits = st.slider("Platform Visits Per Month", 0, 100, 20)
assignment_rate = st.slider("Assignment Completion Rate (%)", 0, 100, 85)

st.divider()

if st.button("🔍 Predict", use_container_width=True):
    input_data = np.array([[age, study_hours, courses_completed, platform_visits, assignment_rate]])
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0]

    if prediction == 1:
        st.success(f"✅ **Student is LIKELY to Purchase a Course!**")
    else:
        st.error(f"❌ **Student is NOT Likely to Purchase a Course.**")

    st.info(f"Confidence: **{max(probability)*100:.1f}%**")

st.divider()
st.caption("Built with ❤️ for Learn Depth Internship | Model: Random Forest")
