import streamlit as st
import pandas as pd
import joblib

# Load the trained model (adjust the path if needed)
model_filename = "behavioral_matching_model.joblib"
model = joblib.load(model_filename)

st.title("Candidate Behavioral Matching Predictor")
st.write("Input candidate and employer engagement data to predict if the candidate is a good match.")

# --- Predefined Test Cases ---
test_cases = {
    "Test Case 1: High engagement": {
        "candidate_profile": "Candidate 001: Enthusiastic about data science and AI, actively engages on multiple platforms.",
        "internship_text": "Internship in Data Science at TechCorp, focusing on innovative machine learning projects.",
        "job_types": "Data Science, Machine Learning, AI",
        "career_interest": "Passionate about tech innovation and research in AI.",
        "job_search_frequency": 15,
        "time_on_site": 45,
        "application_patterns": 10,
        "platform_interaction": 9,
        "candidate_responsiveness": 90,
        "employer_responsiveness": 80,
        "interest_variance": 8,
        "platform_activity_time": 240,
        "click_through_rate": 70,
        "frequency_of_logins": 10,
        "employer_response_time": 30,
        "interview_rate": 50,
        "offer_acceptance_rate": 80,
        "engagement_trend": "increasing"
    },
    "Test Case 2: Low engagement": {
        "candidate_profile": "Candidate 002: Minimal activity with little interest; rarely logs in and seldom searches for jobs.",
        "internship_text": "Internship in Software Engineering at Startup, involving basic tasks with minimal guidance.",
        "job_types": "Software Engineering",
        "career_interest": "Looking for any tech opportunity with minimal commitment.",
        "job_search_frequency": 2,
        "time_on_site": 5,
        "application_patterns": 1,
        "platform_interaction": 2,
        "candidate_responsiveness": 20,
        "employer_responsiveness": 50,
        "interest_variance": 3,
        "platform_activity_time": 30,
        "click_through_rate": 10,
        "frequency_of_logins": 1,
        "employer_response_time": 120,
        "interview_rate": 10,
        "offer_acceptance_rate": 30,
        "engagement_trend": "decreasing"
    },
    "Test Case 3: Mixed signals": {
        "candidate_profile": "Candidate 003: Shows interest in some roles but lacks consistent engagement.",
        "internship_text": "Internship in Marketing at GlobalCorp with varied responsibilities and dynamic projects.",
        "job_types": "Marketing, Sales, Niche Roles",
        "career_interest": "Interested in digital marketing trends and innovative sales strategies.",
        "job_search_frequency": 8,
        "time_on_site": 20,
        "application_patterns": 4,
        "platform_interaction": 5,
        "candidate_responsiveness": 60,
        "employer_responsiveness": 70,
        "interest_variance": 5,
        "platform_activity_time": 120,
        "click_through_rate": 40,
        "frequency_of_logins": 5,
        "employer_response_time": 80,
        "interview_rate": 30,
        "offer_acceptance_rate": 50,
        "engagement_trend": "stable"
    }
}

selected_test_case = st.sidebar.selectbox("Select a predefined test case", options=list(test_cases.keys()))
test_data = test_cases[selected_test_case]
st.sidebar.markdown("### Test Case Details")
st.sidebar.write(test_data)

# --- Input Components ---
# Text inputs
candidate_profile = st.text_area("Candidate Profile", value=test_data["candidate_profile"], height=100)
internship_text = st.text_area("Internship Details", value=test_data["internship_text"], height=100)
job_types = st.text_input("Job Types (comma separated)", value=test_data["job_types"])
career_interest = st.text_area("Career Interest", value=test_data["career_interest"], height=100)

# Engagement Metrics & Numeric Inputs
job_search_frequency = st.number_input("Job Search Frequency (per week)", min_value=0, max_value=100, value=test_data["job_search_frequency"], step=1)
time_on_site = st.number_input("Time on Site (minutes)", min_value=0, max_value=120, value=test_data["time_on_site"], step=1)
application_patterns = st.number_input("Applications per Week", min_value=0, max_value=50, value=test_data["application_patterns"], step=1)
platform_interaction = st.slider("Platform Interaction Score (1-10)", min_value=1, max_value=10, value=test_data["platform_interaction"])

candidate_responsiveness = st.slider("Candidate Responsiveness (%)", min_value=0, max_value=100, value=test_data["candidate_responsiveness"])
employer_responsiveness = st.slider("Employer Responsiveness (%)", min_value=0, max_value=100, value=test_data["employer_responsiveness"])
interest_variance = st.slider("Interest Variance (selectivity 1-10)", min_value=1, max_value=10, value=test_data["interest_variance"])

# Platform Activity
platform_activity_time = st.number_input("Platform Activity Time (minutes per day)", min_value=0, max_value=600, value=test_data["platform_activity_time"], step=1)
click_through_rate = st.slider("Click Through Rate (%)", min_value=0, max_value=100, value=test_data["click_through_rate"])
frequency_of_logins = st.number_input("Frequency of Logins (per week)", min_value=0, max_value=30, value=test_data["frequency_of_logins"], step=1)
# Composite platform activity (simple sum)
platform_activity = platform_activity_time + click_through_rate + frequency_of_logins

# Employer Interaction Data
employer_response_time = st.number_input("Employer Response Time (minutes)", min_value=1, max_value=1440, value=test_data["employer_response_time"], step=1)
interview_rate = st.slider("Interview Rate (%)", min_value=0, max_value=100, value=test_data["interview_rate"])
offer_acceptance_rate = st.slider("Offer Acceptance Rate (%)", min_value=0, max_value=100, value=test_data["offer_acceptance_rate"])
# Composite employer interaction (simple sum)
employer_interaction = employer_response_time + interview_rate + offer_acceptance_rate

# Engagement Trend
engagement_trend_choice = st.selectbox("Engagement Trend", options=["increasing", "stable", "decreasing"],
                                       index=["increasing", "stable", "decreasing"].index(test_data["engagement_trend"]))
if engagement_trend_choice == "increasing":
    engagement_trend = 1
elif engagement_trend_choice == "stable":
    engagement_trend = 0
else:
    engagement_trend = -1

st.markdown("---")
if st.button("Predict Match"):
    # Create the combined text field (required by the model's text processing)
    combined_text = f"{candidate_profile} {internship_text} {job_types} {career_interest}"
    
    # Construct a dictionary with all expected feature columns
    custom_data = {
        "candidate_profile": candidate_profile,
        "internship_text": internship_text,
        "job_types": job_types,
        "career_interest": career_interest,
        "combined_text": combined_text,
        "job_search_frequency": job_search_frequency,
        "time_on_site": time_on_site,
        "application_patterns": application_patterns,
        "platform_interaction": platform_interaction,
        "candidate_responsiveness": candidate_responsiveness,
        "employer_responsiveness": employer_responsiveness,
        "interest_variance": interest_variance,
        "platform_activity": platform_activity,
        "employer_interaction": employer_interaction,
        "engagement_trend": engagement_trend
    }
    
    # Create a DataFrame for the model input
    input_df = pd.DataFrame([custom_data])
    
    # Get predictions and prediction probabilities from the model
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)
    
    # Display results
    result_text = "Good Match" if prediction[0] == 1 else "Not a Good Match"
    st.subheader("Prediction")
    st.write(result_text)
    st.subheader("Prediction Probabilities")
    st.write(prediction_proba)
    
    # Option to download the input data as CSV
    csv_data = input_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Input Data as CSV", csv_data, "input_data.csv", "text/csv")
