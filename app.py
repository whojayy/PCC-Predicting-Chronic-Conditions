import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Health Risk Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling - with dark theme matching the screenshot
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: white;
        text-align: center;
        margin-bottom: 1rem;
        padding-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: white;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background-color: #1e1e1e;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .prediction-result {
        font-size: 1.5rem;
        font-weight: bold;
        color: white;
    }
    .risk-high {
        color: #e74c3c;
    }
    .risk-low {
        color: #27ae60;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid #333;
        color: #7f8c8d;
    }
    .stButton button {
        background-color: #3498db;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    /* Sidebar styling to match the dark theme */
    .css-1d391kg {
        background-color: #1e1e2e;
    }
    /* Remove default Streamlit styling */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    /* Custom section headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Load models from the root directory
@st.cache_resource
def load_models():
    return {
        "highbp": joblib.load("model_highbp.pkl"),
        "diabetes": joblib.load("model_diabetes.pkl"),
        "cardio": joblib.load("model_cardio.pkl")
    }

try:
    models = load_models()
    model_highbp = models["highbp"]
    model_diabetes = models["diabetes"]
    model_cardio = models["cardio"]
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# Thresholds
thresholds = {
    "highbp": 0.23,
    "diabetes": 0.24,
    "cardio": 0.24
}

# Feature options
feature_options = {
    "Age Group": ['18 to 34 years', '35 to 49 years', '50 to 64 years', '65 and older'],
    "Sex at Birth": ['Male', 'Female'],
    "Marital Status": [
        'Married/Common-law',
        'Widowed/Divorced/Separated/Single, never married'
    ],
    "Perceived health ": ['Poor', 'Fair', 'Good', 'Very good', 'Excellent'],
    "Perceived mental health ": ['Poor', 'Fair', 'Good', 'Very good', 'Excellent', 'Unknown'],
    "Satisfaction with life in general ": [
        'Very Satisfied', 'Satisfied', 'Neither satisfied nor dissatisfied', 'Dissatisfied', 'Unknown'
    ],
    "Smoking status": [
        'Non-smoker (abstainer or experimental)',
        'Former daily smoker (non-smoker now)',
        'Current daily smoker',
        'Former occasional smoker (non-smoker now)',
        'Current occasional smoker',
        'Unknown'
    ],
    "Used cannabis - 12 months": ['No', 'Yes', 'Unknown'],
    "Severity of Canabis Dependence": [
        'No cannabis use',
        'Takes cannabis but no dependence',
        'Takes cannabis & dependent on it',
        'Unknown'
    ],
    "Type of drinker": [
        'Regular drinker', 'Occasional drinker', 'Did not drink in the last 12 months'
    ],
    "Drank 5+ / 4+ drinks one occasion - frequency - 12 months": [
        'Never', 'Less than once a month', 'Once a month', '2-3 times a month', 'Once a week', 'More than once a week', 'Valid skip'
    ],
    "Total Household Income - All Sources": [
        '$80,000 or more', '$60,000 to $79,999', '$40,000 to $59,999', '$20,000 to $39,999', 'No income or less than $20,000', 'Unknown'
    ],
    "BMI classification for adults aged 18 and over (adjusted) - international": [
        'Underweight/ Normal weight', 'Overweight / Obese - Class I, II, III', 'Unknown'
    ],
    "Pain health status": ['Has usual pain or discomfort', 'No usual pain or discomfort'],
    "Has sleep apnea": ['No', 'Yes'],
    "Has high blood cholesterol / lipids": ['No', 'Yes', 'Unknown'],
    "High blood cholesterol / lipids - took medication - 1 month": ['No', 'Yes'],
    "Has chronic fatigue syndrome": ['No', 'Yes'],
    "Has a mood disorder (depression, bipolar, mania, dysthymia)": ['No', 'Yes'],
    "Has an anxiety disorder (phobia, OCD, panic)": ['No', 'Yes'],
    "Has respiratory chronic condition (asthma or COPD)": ['No', 'Yes', 'Unknown'],
    "Musculoskeletal condition (Arthritis, fibromyalgia, osteoporosis)": ['No', 'Yes', 'Unknown'],
    "Had a seasonal flu shot (excluding H1N1) - lifetime": ['No', 'Yes', 'Unknown'],
    "Seasonal flu shot - last time": [
        'Less than 1 year ago', '1 year to less than 2 years ago', '2 years ago or more', 'Valid skip', 'Unknown'
    ],
    "Usual place for immediate care for minor problem": ['Yes', 'No'],
    "Considered suicide - lifetime": ['No', 'Yes', 'Unknown'],
    "Considered suicide - last 12 months": ['No', 'Yes', 'Unknown'],
    "High blood pressure - took medication - 1 month": ['No', 'Yes'],
}

# Function to create a gauge chart for risk visualization
def create_gauge_chart(probability, threshold):
    fig, ax = plt.subplots(figsize=(4, 3), subplot_kw={'projection': 'polar'})
    fig.patch.set_facecolor('#1e1e1e')  # Dark background to match theme
    
    # Define the gauge
    theta = np.linspace(0, 180, 100) * np.pi / 180
    r = [1] * 100
    
    # Define colors for different risk levels
    low_risk = np.linspace(0, threshold * 180, int(threshold * 100)) * np.pi / 180
    high_risk = np.linspace(threshold * 180, 180, 100 - int(threshold * 100)) * np.pi / 180
    
    # Plot the gauge background
    ax.plot(low_risk, [1] * len(low_risk), color='green', linewidth=20, alpha=0.6)
    ax.plot(high_risk, [1] * len(high_risk), color='red', linewidth=20, alpha=0.6)
    
    # Plot the needle
    needle_angle = probability * np.pi
    ax.plot([0, needle_angle], [0, 0.8], color='white', linewidth=2)
    ax.scatter(needle_angle, 0.8, color='white', s=100, zorder=10)
    
    # Customize the plot
    ax.set_rticks([])
    ax.set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
    ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'], color='white')
    ax.spines['polar'].set_visible(False)
    ax.grid(False)
    ax.set_facecolor('#1e1e1e')
    
    # Set the limits
    ax.set_ylim(0, 1.2)
    ax.set_title(f'Risk: {probability:.1%}', fontsize=14, color='white')
    
    return fig

# Create sidebar with information - matching the screenshot
with st.sidebar:
    st.title("About This Tool")
    st.markdown("### How It Works")
    st.markdown("""
    This health screening tool uses machine 
    learning models trained on the Canadian 
    Community Health Survey (CCHS) dataset 
    to predict your risk for:
    
    1. High Blood Pressure
    2. Diabetes
    3. Cardiovascular Disease
    """)
    
    st.markdown("### Interpretation")
    st.markdown("""
    - The tool provides probability scores for each condition
    - Higher scores indicate greater risk
    - This is a screening tool, not a diagnostic tool
    - Always consult healthcare professionals for medical advice
    """)
    
    st.markdown("### Privacy")
    st.markdown("""
    Your data is processed locally and is not 
    stored or shared.
    """)
    
    st.markdown("© 2025 PCC Project")

# Header with hospital icon
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    st.markdown('<h1 class="main-header">🏥 Progressive Health Risk Predictor</h1>', unsafe_allow_html=True)

# Removed the blue info box as requested
# Just adding a simple text description instead
st.write("This tool predicts your risk for three chronic health conditions based on your lifestyle and health information. The predictions follow a progressive model: High Blood Pressure → Diabetes → Cardiovascular Disease.")

# Main content area
tab1, tab2 = st.tabs(["Risk Assessment", "About the Models"])

with tab1:
    # Input Form with better organization
    with st.form("user_input_form"):
        st.markdown('<h2 class="section-header">Lifestyle & Health Information</h2>', unsafe_allow_html=True)
        
        # Organize form into sections
        st.markdown('<div class="section-header">Demographics</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            input_dict = {}
            input_dict["Age Group"] = st.selectbox("Age Group", feature_options["Age Group"])
        with col2:
            input_dict["Sex at Birth"] = st.selectbox("Sex at Birth", feature_options["Sex at Birth"])
        with col3:
            input_dict["Marital Status"] = st.selectbox("Marital Status", feature_options["Marital Status"])
        
        st.markdown('<div class="section-header">Health Perception</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            input_dict["Perceived health "] = st.selectbox("Perceived Physical Health", feature_options["Perceived health "])
            input_dict["Pain health status"] = st.selectbox("Pain Status", feature_options["Pain health status"])
        with col2:
            input_dict["Perceived mental health "] = st.selectbox("Perceived Mental Health", feature_options["Perceived mental health "])
            input_dict["Satisfaction with life in general "] = st.selectbox("Life Satisfaction", feature_options["Satisfaction with life in general "])
        
        st.markdown('<div class="section-header">Lifestyle Factors</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            input_dict["Smoking status"] = st.selectbox("Smoking Status", feature_options["Smoking status"])
            input_dict["Type of drinker"] = st.selectbox("Alcohol Consumption", feature_options["Type of drinker"])
        with col2:
            input_dict["Used cannabis - 12 months"] = st.selectbox("Cannabis Use (12 months)", feature_options["Used cannabis - 12 months"])
            input_dict["Drank 5+ / 4+ drinks one occasion - frequency - 12 months"] = st.selectbox("Binge Drinking Frequency", 
                                                                                                feature_options["Drank 5+ / 4+ drinks one occasion - frequency - 12 months"])
        
        # Continue with other sections (abbreviated for clarity)
        with st.expander("Body Metrics & Physical Health"):
            input_dict["BMI classification for adults aged 18 and over (adjusted) - international"] = st.selectbox(
                "BMI Classification", 
                feature_options["BMI classification for adults aged 18 and over (adjusted) - international"]
            )
            input_dict["Has sleep apnea"] = st.selectbox("Sleep Apnea", feature_options["Has sleep apnea"])
            input_dict["Has respiratory chronic condition (asthma or COPD)"] = st.selectbox(
                "Respiratory Condition", 
                feature_options["Has respiratory chronic condition (asthma or COPD)"]
            )
            input_dict["Musculoskeletal condition (Arthritis, fibromyalgia, osteoporosis)"] = st.selectbox(
                "Musculoskeletal Condition", 
                feature_options["Musculoskeletal condition (Arthritis, fibromyalgia, osteoporosis)"]
            )
        
        with st.expander("Mental Health"):
            input_dict["Has a mood disorder (depression, bipolar, mania, dysthymia)"] = st.selectbox(
                "Mood Disorder", 
                feature_options["Has a mood disorder (depression, bipolar, mania, dysthymia)"]
            )
            input_dict["Has an anxiety disorder (phobia, OCD, panic)"] = st.selectbox(
                "Anxiety Disorder", 
                feature_options["Has an anxiety disorder (phobia, OCD, panic)"]
            )
            input_dict["Considered suicide - lifetime"] = st.selectbox("Suicidal Thoughts (Lifetime)", feature_options["Considered suicide - lifetime"])
            input_dict["Considered suicide - last 12 months"] = st.selectbox("Suicidal Thoughts (12 months)", feature_options["Considered suicide - last 12 months"])
        
        with st.expander("Healthcare & Socioeconomic Factors"):
            input_dict["Total Household Income - All Sources"] = st.selectbox("Household Income", feature_options["Total Household Income - All Sources"])
            input_dict["Usual place for immediate care for minor problem"] = st.selectbox(
                "Regular Healthcare Access", 
                feature_options["Usual place for immediate care for minor problem"]
            )
            input_dict["Had a seasonal flu shot (excluding H1N1) - lifetime"] = st.selectbox(
                "Flu Shot (Ever)", 
                feature_options["Had a seasonal flu shot (excluding H1N1) - lifetime"]
            )
            input_dict["Seasonal flu shot - last time"] = st.selectbox("Last Flu Shot", feature_options["Seasonal flu shot - last time"])
        
        with st.expander("Cholesterol & Fatigue"):
            input_dict["Has high blood cholesterol / lipids"] = st.selectbox("High Cholesterol", feature_options["Has high blood cholesterol / lipids"])
            input_dict["High blood cholesterol / lipids - took medication - 1 month"] = st.selectbox(
                "Cholesterol Medication", 
                feature_options["High blood cholesterol / lipids - took medication - 1 month"]
            )
            input_dict["Has chronic fatigue syndrome"] = st.selectbox("Chronic Fatigue", feature_options["Has chronic fatigue syndrome"])
            input_dict["Severity of Canabis Dependence"] = st.selectbox("Cannabis Dependence", feature_options["Severity of Canabis Dependence"])
        
        # Known conditions section
        st.markdown('<div class="section-header">Known Conditions</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        
        with col1:
            bp_known = st.radio("Do you already know if you have high blood pressure?", ["No", "Yes"])
            if bp_known == "Yes":
                bp_val = st.selectbox("High Blood Pressure Status", ["No", "Yes"])
                input_dict['Has a high blood pressure'] = bp_val
                input_dict["High blood pressure - took medication - 1 month"] = st.selectbox(
                    "Blood Pressure Medication", 
                    feature_options["High blood pressure - took medication - 1 month"]
                )
            else:
                input_dict["High blood pressure - took medication - 1 month"] = "No"
        
        with col2:
            diab_known = st.radio("Do you already know if you have diabetes?", ["No", "Yes"])
            if diab_known == "Yes":
                diab_val = st.selectbox("Diabetes Status", ["No", "Yes"])
                input_dict['Has diabetes'] = diab_val
        
        # Submit button with better styling
        submit = st.form_submit_button("🔍 Predict My Health Risks")

    # Results section
    if submit:
        st.markdown('<h2 class="section-header">Risk Assessment Results</h2>', unsafe_allow_html=True)
        
        # Create a DataFrame from the input
        user_df = pd.DataFrame([input_dict])
        
        # Create three columns for the three conditions
        col1, col2, col3 = st.columns(3)
        
        # --- High Blood Pressure Prediction ---
        with col1:
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.markdown('<h3 style="text-align: center;">High Blood Pressure</h3>', unsafe_allow_html=True)
            
            if 'Has a high blood pressure' not in user_df.columns:
                proba = model_highbp.predict_proba(user_df)[0][1]
                prediction = "Yes" if proba >= thresholds['highbp'] else "No"
                user_df['Has a high blood pressure'] = prediction
                
                # Display gauge chart
                st.pyplot(create_gauge_chart(proba, thresholds['highbp']))
                
                # Display prediction with color coding
                risk_class = "risk-high" if prediction == "Yes" else "risk-low"
                st.markdown(f'<p class="prediction-result">Prediction: <span class="{risk_class}">{prediction}</span></p>', unsafe_allow_html=True)
                st.markdown(f"Probability: {proba:.2f}")
                
                # Progress bar for visual representation
                st.progress(proba)
            else:
                st.markdown(f'<p class="prediction-result">Known Status: {user_df["Has a high blood pressure"].iloc[0]}</p>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # --- Diabetes Prediction ---
        with col2:
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.markdown('<h3 style="text-align: center;">Diabetes</h3>', unsafe_allow_html=True)
            
            if 'Has diabetes' not in user_df.columns:
                proba = model_diabetes.predict_proba(user_df)[0][1]
                prediction = "Yes" if proba >= thresholds['diabetes'] else "No"
                user_df['Has diabetes'] = prediction
                
                # Display gauge chart
                st.pyplot(create_gauge_chart(proba, thresholds['diabetes']))
                
                # Display prediction with color coding
                risk_class = "risk-high" if prediction == "Yes" else "risk-low"
                st.markdown(f'<p class="prediction-result">Prediction: <span class="{risk_class}">{prediction}</span></p>', unsafe_allow_html=True)
                st.markdown(f"Probability: {proba:.2f}")
                
                # Progress bar for visual representation
                st.progress(proba)
            else:
                st.markdown(f'<p class="prediction-result">Known Status: {user_df["Has diabetes"].iloc[0]}</p>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # --- Cardiovascular Prediction ---
        with col3:
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.markdown('<h3 style="text-align: center;">Cardiovascular Disease</h3>', unsafe_allow_html=True)
            
            cardio_proba = model_cardio.predict_proba(user_df)[0][1]
            cardio_pred = "Yes" if cardio_proba >= thresholds['cardio'] else "No"
            
            # Display gauge chart
            st.pyplot(create_gauge_chart(cardio_proba, thresholds['cardio']))
            
            # Display prediction with color coding
            risk_class = "risk-high" if cardio_pred == "Yes" else "risk-low"
            st.markdown(f'<p class="prediction-result">Prediction: <span class="{risk_class}">{cardio_pred}</span></p>', unsafe_allow_html=True)
            st.markdown(f"Probability: {cardio_proba:.2f}")
            
            # Progress bar for visual representation
            st.progress(cardio_proba)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Simple disclaimer without the blue box
        st.write("**Disclaimer:** This tool provides risk estimates based on statistical models and should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider for medical concerns.")

with tab2:
    st.markdown("""
    ## About the Prediction Models
    
    ### Data Source
    The models were trained using the Canadian Community Health Survey (CCHS) 2019-2020 dataset, which includes health information from thousands of respondents across Canada.
    
    ### Model Type
    All three predictions use Logistic Regression models, which were selected for their balance of performance and interpretability.
    
    ### Performance Metrics
    
    | Condition | ROC-AUC | Recall (Default) | Recall (Optimized) |
    |-----------|---------|-----------------|-------------------|
    | High BP | 0.810 | 0.785 | 0.960 |
    | Diabetes | 0.853 | 0.803 | 0.960 |
    | Cardiovascular | 0.856 | 0.814 | 0.950 |
    
    ### Threshold Optimization
    The models use custom thresholds optimized using Youden's J statistic to maximize recall (sensitivity) while maintaining acceptable precision.
    
    ### Progressive Prediction Logic
    The system follows clinical progression patterns:
    1. High Blood Pressure
    2. Diabetes
    3. Cardiovascular Disease
    
    This approach reflects how these conditions often develop sequentially in real life.
    """)

# Footer
st.markdown('<div class="footer">Developed for the PCC: Predicting Chronic Conditions Project</div>', unsafe_allow_html=True)