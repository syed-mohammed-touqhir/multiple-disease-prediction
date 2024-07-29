import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load datasets
dataset = pd.read_csv('Training.csv')
sym_des = pd.read_csv("symtoms_df.csv")
precautions = pd.read_csv("precautions_df.csv")
workout = pd.read_csv("workout_df.csv")
description = pd.read_csv("description.csv")
medications = pd.read_csv('medications.csv')
diets = pd.read_csv("diets.csv")

# Prepare data
X = dataset.drop('prognosis', axis=1)
y = dataset['prognosis']
le = LabelEncoder()
le.fit(y)
Y = le.transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=20)

# Train SVC model
svc = SVC(kernel='linear')
svc.fit(X_train, y_train)

# Save and load model
pickle.dump(svc, open('svc.pkl', 'wb'))
svc = pickle.load(open('svc.pkl', 'rb'))

# Define symptoms_dict (partial list)
symptoms_dict = {
    'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3,
    'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8,
    'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12,
    'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16,
    'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20,
    'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24,
    'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28,
    'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32,
    'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36,
    'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40,
    'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44,
    'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47,
    'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51,
    'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55,
    'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58,
    'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61,
    'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65,
    'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69,
    'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72,
    'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75,
    'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78,
    'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82,
    'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85,
    'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88,
    'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91,
    'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94,
    'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98,
    'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101,
    'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104,
    'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108,
    'lack_of_concentration': 109, 'visual_disturbances': 110,
    'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112,
    'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115,
    'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118,
    'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121,
    'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125,
    'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128,
    'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131
}

# Define diseases_list (updated)
diseases_list = {
    0: '(vertigo) Paroymsal  Positional Vertigo', 1: 'AIDS', 2: 'Acne', 3: 'Alcoholic hepatitis',
    4: 'Allergy', 5: 'Arthritis', 6: 'Bronchial Asthma', 7: 'Cervical spondylosis', 8: 'Chicken pox',
    9: 'Chronic cholestasis', 10: 'Common Cold', 11: 'Dengue', 12: 'Diabetes ', 13: 'Dimorphic hemmorhoids(piles)',
    14: 'Drug Reaction', 15: 'Fungal infection', 16: 'GERD', 17: 'Gastroenteritis', 18: 'Heart attack',
    19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 23: 'Hypertension ',
    24: 'Hyperthyroidism', 25: 'Hypoglycemia', 26: 'Hypothyroidism', 27: 'Impetigo', 28: 'Jaundice',
    29: 'Malaria', 30: 'Migraine', 31: 'Osteoarthristis', 32: 'Paralysis (brain hemorrhage)', 33: 'Peptic ulcer diseae',
    34: 'Pneumonia', 35: 'Psoriasis', 36: 'Tuberculosis', 37: 'Typhoid', 38: 'Urinary tract infection',
    39: 'Varicose veins'
}

# Define helper function
def helper(dis):
    # Description
    desc = description[description['Disease'] == dis]['Description']
    desc = desc.values[0] if not desc.empty else "No description available"

    # Precautions
    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']]
    pre = [col for col in pre.values.flatten() if pd.notna(col)] if not pre.empty else ["No precautions available"]

    # Medications
    med = medications[medications['Disease'] == dis]['Medication']
    med = [med for med in med.values if pd.notna(med)] if not med.empty else ["No medications available"]

    # Diets
    die = diets[diets['Disease'] == dis]['Diet']
    die = [die for die in die.values if pd.notna(die)] if not die.empty else ["No diets available"]

    # Workout
    wrkout = workout[workout['disease'] == dis]['workout']
    wrkout = [wrk for wrk in wrkout.values if pd.notna(wrk)] if not wrkout.empty else ["No workouts available"]

    # Debug output
    print(f"Debug: Description found: {desc}")
    print(f"Debug: Precautions found: {pre}")
    print(f"Debug: Medications found: {med}")
    print(f"Debug: Diets found: {die}")
    print(f"Debug: Workouts found: {wrkout}")

    return desc, pre, med, die, wrkout

# Encoding function
def encode_symptoms(symptoms):
    # Create a feature vector of zeros with length of all possible symptoms
    feature_vector = np.zeros(len(symptoms_dict), dtype=int)
    for symptom in symptoms:
        if symptom in symptoms_dict:
            feature_vector[symptoms_dict[symptom]] = 1
    return feature_vector.reshape(1, -1)

# Example prediction
while True:
    test_symptoms = input("Enter the symptoms separated by commas: ")
    test_symptoms = [s.strip() for s in test_symptoms.split(',')]
    
    if all(symptom in symptoms_dict for symptom in test_symptoms):
        test_symptoms_encoded = encode_symptoms(test_symptoms)
        break
    else:
        print("Invalid symptom entered. Please enter valid symptoms.")

# Predict on the test set
y_pred = svc.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy * 100:.2f}%")

# Predict disease
predicted = svc.predict(test_symptoms_encoded)
predicted_disease = diseases_list[int(predicted[0])]

# Display results
desc, pre, med, die, wrkout = helper(predicted_disease)

print("================= Predicted Disease =================")
print(predicted_disease)

print("================= Description =======================")
print(desc)

print("================= Precautions =======================")
for i, p_i in enumerate(pre, 1):
    print(f"{i}: {p_i}")

print("================= Medications =======================")
for i, m_i in enumerate(med, 1):
    print(f"{i}: {m_i}")

print("================= Workout ===========================")
for i, w_i in enumerate(wrkout, 1):
    print(f"{i}: {w_i}")

print("================= Diets =============================")
for i, d_i in enumerate(die, 1):
    print(f"{i}: {d_i}")
