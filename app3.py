import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import re
svc = pickle.load(open('svc.pkl', 'rb'))
if not os.path.exists('svc.pkl'):
    raise FileNotFoundError('svc.pkl file not found')
# Load the trained model


# Load datasets
symptoms_dict = {
    'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5,
    'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11,
    'burning_micturition': 12, 'spotting_urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16,
    'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21,
    'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26,
    'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32,
    'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37,
    'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42,
    'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46,
    'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51,
    'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56,
    'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60,
    'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66,
    'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71,
    'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75,
    'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80,
    'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85,
    'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89,
    'foul_smell_of_urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93,
    'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98,
    'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic_patches': 102,
    'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107,
    'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111,
    'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115,
    'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118,
    'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122,
    'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127,
    'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131
}

diseases_list = {
    15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction',
    33: 'Peptic ulcer disease', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma',
    23: 'Hypertension', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)',
    28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'Hepatitis A',
    19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis',
    36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemorrhoids (piles)',
    18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia',
    31: 'Osteoarthritis', 5: 'Arthritis', 0: 'Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis',
    27: 'Impetigo'
}

# Initialize global variables
users_db = {}

# Load additional datasets
sym_des = pd.read_csv("symtoms_df.csv")
precautions = pd.read_csv("precautions_df.csv")
workout = pd.read_csv("workout_df.csv")
description = pd.read_csv("description.csv")
medications = pd.read_csv('medications.csv')
diets = pd.read_csv("diets.csv")

# Helper function
def helper(dis):
    desc = description[description['Disease'] == dis]['Description'].values[0]
    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = pre.values.flatten()
    med = medications[medications['Disease'] == dis]['Medication'].tolist()
    die = diets[diets['Disease'] == dis]['Diet'].tolist()
    wrkout = workout[workout['disease'] == dis]['workout'].tolist()
    return desc, pre, med, die, wrkout

# Model Prediction function
def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        if item in symptoms_dict:
            input_vector[symptoms_dict[item]] = 1
    disease_code = svc.predict([input_vector])[0]
    return diseases_list.get(disease_code, 'Unknown disease')

# CSS for the app
st.markdown("""
    <style>
    /* Background color and text color for the entire app */
    body {
        background-color:#DBDDE2; /* Custom background color */
        color: #202020; /* Text color */
    }
    
    /* Customize button appearance */
    .stButton > button {
        background-color: #007bff; /* Button background color */
        color: #ffffff; /* Button text color */
        width: 200px; /* Button width */
        font-size: 20px; /* Button text size */
        font-weight: bold; /* Make button text bold */
        margin: 0; /* No additional margin */
        border-radius: 15px; /* Rounded corners */
        box-shadow: 0px 4px 6px rgba(0,0,0,0.1); /* Shadow */
        transition: background-color 0.3s ease, box-shadow 0.3s ease, transform 0.3s ease; /* Transition effect */
        display: block; /* Ensure button is a block element */
        text-align: center; /* Center text inside the button */
    }

    /* Hover effects for buttons */
    .stButton > button:hover {
        background-color: #0056b3; /* Darker background on hover */
        box-shadow: 0px 6px 12px rgba(0,0,0,0.2); /* Increased shadow on hover */
        transform: scale(1.05); /* Slightly enlarge button on hover */
    }

    /* Styling for headers */
    .css-18e3th9 {
        color: #202020; /* Black text color for headers */
    }

    /* Styling for markdown text */
    .css-1v3fvcr {
        color: #202020; /* Black text color */
    }
    
    /* Increase font size for text input field */
    .stTextInput input {
        font-size: 20px; /* Increase text size */
        border: 2px solid #007bff; /* Border color */
        border-radius: 10px; /* Rounded corners */
        padding: 10px; /* Padding inside the input field */
    }
    
    /* Increase font size and make output text bold */
    .output-text {
        font-weight: bold; /* Make text bold */
        font-size: 24px; /* Increase font size */
        color: #202020; /* Black text color */
    }
    </style>
""", unsafe_allow_html=True)

# Streamlit app
st.title('Multiple Disease Prediction')

# Sign-up Page
def validate_password(password):
    if len(password) < 8:
        return "Password must be at least 8 characters long."
    if not re.search(r'[A-Z]', password):
        return "Password must contain at least one uppercase letter."
    if not re.search(r'[a-z]', password):
        return "Password must contain at least one lowercase letter."
    if not re.search(r'[0-9]', password):
        return "Password must contain at least one digit."
    if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
        return "Password must contain at least one special character."
    return None

def sign_up_page():
    st.header('Sign Up')
    username = st.text_input('Username')
    password = st.text_input('Password', type='password')
    confirm_password = st.text_input('Confirm Password', type='password')

    if st.button('Sign Up'):
        if username in users_db:
            st.error('Username already exists. Please choose a different username.')
        elif password != confirm_password:
            st.error('Passwords do not match. Please try again.')
        elif not username or not password:
            st.warning('Please enter both username and password.')
        else:
            password_error = validate_password(password)
            if password_error:
                st.error(password_error)
            else:
                users_db[username] = password
                st.session_state.signed_up = True
                st.success('Sign up successful! Redirecting to the main page...')
                st.experimental_rerun()  # Redirect to the main page

def main_page():
    st.header('Enter your symptoms')

    # Create a multiselect widget for symptoms
    selected_symptoms = st.multiselect(
        "Select your symptoms", 
        options=list(symptoms_dict.keys()),
        default=None
    )

    # User input
    symptoms_input = ', '.join(selected_symptoms)

    # Predict button
    if st.button('Predict'):
        if symptoms_input:
            user_symptoms = [s.strip() for s in symptoms_input.split(',')]
            predicted_disease = get_predicted_value(user_symptoms)
            st.session_state.predicted_disease = predicted_disease
            st.session_state.show_info = 'None'
        else:
            st.write('Please enter symptoms.')

    if 'predicted_disease' in st.session_state:
        st.markdown('<br><br>', unsafe_allow_html=True)
        
        with st.container():
            col1, col2 = st.columns([1, 1])
            col3, col4 = st.columns([1, 1])
            col5, col6 = st.columns([1, 1])
        
            with col1:
                if st.button('Disease'):
                    st.session_state.show_info = 'Disease'
        
            with col2:
                if st.button('Description'):
                    st.session_state.show_info = 'Description'
        
            with col3:
                if st.button('Precautions'):
                    st.session_state.show_info = 'Precaution'
        
            with col4:
                if st.button('Medications'):
                    st.session_state.show_info = 'Medications'
        
            with col5:
                if st.button('Workouts'):
                    st.session_state.show_info = 'Workouts'
        
            with col6:
                if st.button('Diets'):
                    st.session_state.show_info = 'Diets'
        
        if 'show_info' in st.session_state:
            if st.session_state.show_info == 'Disease':
                st.subheader('Predicted Disease')
                st.markdown(f'<p class="output-text">{st.session_state.predicted_disease}</p>', unsafe_allow_html=True)
            
            elif st.session_state.show_info == 'Description':
                desc, _, _, _, _ = helper(st.session_state.predicted_disease)
                st.subheader('Description')
                st.markdown(f'<p class="output-text">{desc}</p>', unsafe_allow_html=True)
            
            elif st.session_state.show_info == 'Precaution':
                _, pre, _, _, _ = helper(st.session_state.predicted_disease)
                st.subheader('Precautions')
                for i, p in enumerate(pre, start=1):
                    st.markdown(f'<p class="output-text">{i}: {p}</p>', unsafe_allow_html=True)
            
            elif st.session_state.show_info == 'Medications':
                _, _, med, _, _ = helper(st.session_state.predicted_disease)
                st.subheader('Medications')
                for i, m in enumerate(med, start=1):
                    st.markdown(f'<p class="output-text">{i}: {m}</p>', unsafe_allow_html=True)
            
            elif st.session_state.show_info == 'Workouts':
                _, _, _, _, wrkout = helper(st.session_state.predicted_disease)
                st.subheader('Workouts')
                for i, w in enumerate(wrkout, start=1):
                    st.markdown(f'<p class="output-text">{i}: {w}</p>', unsafe_allow_html=True)
            
            elif st.session_state.show_info == 'Diets':
                _, _, _, die, _ = helper(st.session_state.predicted_disease)
                st.subheader('Diets')
                for i, d in enumerate(die, start=1):
                    st.markdown(f'<p class="output-text">{i}: {d}</p>', unsafe_allow_html=True)
    else:
        st.write('Please click "Predict" first to get the results.')

# Initialize session state
if 'signed_up' not in st.session_state:
    st.session_state.signed_up = False

# Page navigation
pages = {
    "Sign Up": sign_up_page,
    "Main": main_page,
}

# Select page based on sign-up status
if st.session_state.signed_up:
    page = "Main"
else:
    page = "Sign Up"

# Display selected page
pages[page]() 
