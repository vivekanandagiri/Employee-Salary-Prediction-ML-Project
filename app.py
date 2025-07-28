import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- Page Configuration ---
st.set_page_config(
    page_title="Employee Salary Predictor",
    page_icon="🤖",
    layout="wide"
)

# --- Load Model ---
@st.cache_resource
def load_model_and_features():
    """
    Loads the trained model and extracts the feature names it was trained on.
    This function is cached so it only runs once.
    """
    try:
        with open('best_model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        # Extract the feature names the model was trained on
        model_features = model.feature_names_in_
        return model, model_features
    except FileNotFoundError:
        st.error("Error: `best_model.pkl` not found. Please make sure the model file is in the same directory.")
        return None, None
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        return None, None

model, model_features = load_model_and_features()

if model is None:
    st.stop()

# --- Hardcoded values for UI dropdowns (as provided in your script) ---
# These should match the values used during model training
ordinal_features_map = {
    'experience_level': ['EN( Entry-level )', 'MI( Mid-level )', 'SE( Senior )', 'EX( Executive )'],
    'education_required': ['Associate', 'Bachelor', 'Master', 'PhD'],
    'company_size': ['S( Small )', 'M( Medium )', 'L( Large )']
}

unique_values = {
    'job_title': ['AI Research Scientist', 'AI Software Engineer', 'AI Specialist', 'NLP Engineer', 'AI Consultant', 'AI Architect', 'Principal Data Scientist', 'Data Analyst', 'Autonomous Systems Engineer', 'AI Product Manager', 'Machine Learning Engineer', 'Data Engineer', 'Research Scientist', 'ML Ops Engineer', 'Robotics Engineer', 'Head of AI', 'Deep Learning Engineer', 'Data Scientist', 'Machine Learning Researcher', 'Computer Vision Engineer'],
    'company_location': ['China', 'Canada', 'Switzerland', 'India', 'France', 'Germany', 'United Kingdom', 'Singapore', 'Austria', 'Sweden', 'South Korea', 'Norway', 'Netherlands', 'United States', 'Israel', 'Australia', 'Ireland', 'Denmark', 'Finland', 'Japan'],
    'employment_type': ['CT', 'FL', 'PT', 'FT'],
    'industry': ['Automotive', 'Media', 'Education', 'Consulting', 'Healthcare', 'Gaming', 'Government', 'Telecommunications', 'Manufacturing', 'Energy', 'Technology', 'Real Estate', 'Finance', 'Transportation', 'Retail'],
    'company_name': ['Smart Analytics', 'TechCorp Inc', 'Autonomous Tech', 'Future Systems', 'Advanced Robotics', 'Neural Networks Co', 'DataVision Ltd', 'Cloud AI Solutions', 'Quantum Computing Inc', 'Predictive Systems', 'AI Innovations', 'Algorithmic Solutions', 'Cognitive Computing', 'DeepTech Ventures', 'Machine Intelligence Group', 'Digital Transformation LLC'],
    'remote_ratio': [0, 50, 100]
}


# --- App Header ---
st.title('🤖 Employee  Salary Predictor')
st.markdown("This app predicts the annual salary (in USD) for AI and Machine Learning roles. Fill in the details below to get an estimate.")
st.markdown("---")

# --- Input Fields ---
st.subheader("Enter Job Details")

# Create a two-column layout for a cleaner UI
col1, col2 = st.columns(2)

with col1:
    job_title = st.selectbox('Job Title', options=unique_values['job_title'])
    experience_level = st.selectbox('Experience Level', options=ordinal_features_map['experience_level'])
    education_required = st.selectbox('Education Required', options=ordinal_features_map['education_required'])
    years_experience = st.number_input('Years of Experience', min_value=0, max_value=40, value=5)
    
with col2:
    company_location = st.selectbox('Company Location', options=unique_values['company_location'])
    company_size = st.selectbox('Company Size', options=ordinal_features_map['company_size'])
    industry = st.selectbox('Industry', options=unique_values['industry'])
    remote_ratio = st.selectbox('Remote Ratio (%)', options=unique_values['remote_ratio'])


# --- Prediction Logic ---
if st.button('**Predict Salary**', use_container_width=True):
    try:
        # Create a dataframe with all the model's expected features, initialized to 0
        input_data = pd.DataFrame(0, index=[0], columns=model_features)

        # --- Manually encode the user's input ---
        
        # 1. Set numerical features directly
        input_data['years_experience'] = years_experience
        
        # 2. Set ordinal features by finding the index in the list
        input_data['experience_level'] = ordinal_features_map['experience_level'].index(experience_level)
        input_data['education_required'] = ordinal_features_map['education_required'].index(education_required)
        input_data['company_size'] = ordinal_features_map['company_size'].index(company_size)
        
        # 3. Set one-hot encoded features by finding the correct column and setting it to 1
        # Handle company location
        loc_feature = f"company_location_{company_location}"
        if loc_feature in input_data.columns:
            input_data[loc_feature] = 1
            
        # Handle industry
        ind_feature = f"industry_{industry}"
        if ind_feature in input_data.columns:
            input_data[ind_feature] = 1

        # Handle remote ratio
        remote_feature = f"remote_ratio_{remote_ratio}"
        if remote_feature in input_data.columns:
            input_data[remote_feature] = 1
            
        # Handle job title
        job_feature = f"job_title_{job_title}"
        if job_feature in input_data.columns:
            input_data[job_feature] = 1

        # Predict using the prepared dataframe
        # The model was likely trained on the log of the salary
        predicted_log_salary = model.predict(input_data)
        
        # Convert the prediction back to the actual salary
        predicted_actual_salary = np.expm1(predicted_log_salary[0])

        # --- Display Result ---
        st.markdown("---")
        st.subheader("✨ Predicted Annual Salary (USD)")
        st.markdown(f"<h1 style='text-align: center; color: #28a745;'>${predicted_actual_salary:,.2f}</h1>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.error("Please ensure the selected values are valid and try again.")
