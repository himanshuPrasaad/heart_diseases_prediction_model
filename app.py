import streamlit as st
import pandas as pd
import joblib

st.markdown("""
<style>

/* =========================
   MAIN APP BACKGROUND
   ========================= */
.stApp {
    background: linear-gradient(
        135deg,
        #071A2F 0%,
        #0B2A4A 50%,
        #103B5C 100%
    );
    color: white;
}


/* =========================
   MAIN CONTENT AREA
   ========================= */
.block-container {
    max-width: 900px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}


/* =========================
   MAIN HEADING
   ========================= */
h1 {
    background: linear-gradient(
        90deg,
        #FFFFFF,
        #D9F0FF
    );

    padding: 18px 25px;
    border-radius: 15px;

    text-align: center;

    color: #071A2F !important;
    font-weight: 900 !important;

    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.30);

    margin-bottom: 15px;
}


/* =========================
   DESCRIPTION
   ========================= */
.stMarkdown p {
    color: #D7E8F5 !important;
    font-size: 16px;
}


/* =========================
   INPUT LABELS
   AGE, SEX, ETC.
   ========================= */
[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label,
[data-testid="stWidgetLabel"] span {
    color: #FFFFFF !important;
    font-weight: 700 !important;
    font-size: 15px !important;
}


/* =========================
   NUMBER INPUT BOXES
   ========================= */
div[data-baseweb="input"] > div {
    background-color: #F1F4F8 !important;
    border: 1px solid #D5DCE5 !important;
    border-radius: 10px !important;
}


/* Number input text */
div[data-baseweb="input"] input {
    color: #172B4D !important;
    -webkit-text-fill-color: #172B4D !important;
}


/* Number input +/- buttons */
div[data-baseweb="input"] button {
    color: #172B4D !important;
}


/* =========================
   SELECTBOX
   ========================= */
div[data-baseweb="select"] > div {
    background-color: #F1F4F8 !important;
    border: 1px solid #D5DCE5 !important;
    border-radius: 10px !important;
}


/* Selected value inside selectbox */
div[data-baseweb="select"] div[role="button"] {
    color: #172B4D !important;
}


/* Selectbox text */
div[data-baseweb="select"] span {
    color: #172B4D !important;
}


/* Dropdown arrow */
div[data-baseweb="select"] svg {
    color: #172B4D !important;
    fill: #172B4D !important;
}


/* =========================
   SLIDER
   ========================= */
.stSlider {
    padding-top: 5px;
    padding-bottom: 10px;
}


/* Slider label */
.stSlider label,
.stSlider label p {
    color: white !important;
    font-weight: 700 !important;
}


/* Slider value */
.stSlider [data-baseweb="slider"] div {
    font-weight: 700;
}


/* =========================
   PREDICT BUTTON
   ========================= */
.stButton > button {
    width: 100%;

    background: linear-gradient(
        90deg,
        #1683D8,
        #19A7E8
    );

    color: white !important;

    font-size: 18px;
    font-weight: 800;

    padding: 12px 20px;

    border-radius: 12px;
    border: none;

    box-shadow: 0 6px 18px rgba(0, 0, 0, 0.30);

    transition: all 0.25s ease;
}


/* Button hover */
.stButton > button:hover {
    transform: translateY(-2px);

    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.40);
}


/* =========================
   RESULT MESSAGE
   ========================= */
div[data-testid="stAlert"] {
    border-radius: 12px !important;
    font-weight: 700 !important;
}


/* =========================
   STREAMLIT HEADER
   ========================= */
[data-testid="stHeader"] {
    background-color: transparent;
}


/* =========================
   HIDE FOOTER
   ========================= */
footer {
    visibility: hidden;
}

</style>
""", unsafe_allow_html=True)

model = joblib.load("logistic_heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_column = joblib.load("columns.pkl")

st.title("Heart Disease Prediction By Himanshu".upper())
st.markdown("Provide the following details")

age = st.slider("AGE",18,100,40)
sex = st.selectbox("SEX", ['Male','Female'])
chest_pain = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
resting_bp = st.number_input("RESTING BLOOD PRESSURE (mm/Hg)",80,200)
cholesterol = st.number_input("CHOLESTEROL",100,600)
fasting_bs = st.selectbox("FASTING BLOOD SUGAR > 120 mg/dL",[0,1])
resting_ecg = st.selectbox("RESTING ECG", ["Normal","ST-T Wave Abnormality","Left Ventricular Hypertrophy (LVH)"])
max_hr = st.slider("MAX HEART RATE",60,220,150)
exercise_angina = st.selectbox("EXERCISE INDUCED ANGINA",["No","Yes"])
oldpeak = st.number_input("OLDPEAK (ST Depression)",-2.6,6.0,1.0)
st_slope = st.selectbox("ST SLOPE", ["Upsloping","Flat","Downsloping"])

if st.button("Predict"):

    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,

        'Sex_M': int(sex == 'Male'),

        'ChestPainType_ATA': int(chest_pain == 'Atypical Angina'),
        'ChestPainType_NAP': int(chest_pain == 'Non-anginal Pain'),
        'ChestPainType_TA': int(chest_pain == 'Typical Angina'),

        'RestingECG_Normal': int(resting_ecg == 'Normal'),
        'RestingECG_ST': int(resting_ecg == 'ST-T Wave Abnormality'),

        'ExerciseAngina_Y': int(exercise_angina == 'Yes'),

        'ST_Slope_Flat': int(st_slope == 'Flat'),
        'ST_Slope_Up': int(st_slope == 'Upsloping')
    }

    input_df = pd.DataFrame([raw_input])

    input_df = input_df.reindex(
        columns=expected_column,
        fill_value=0
    )

    scaled_input = scaler.transform(input_df)

    prediction = model.predict(scaled_input)[0]

    st.write("Prediction:", prediction)

    if prediction == 1:
        st.error("⚠️ Risk of heart disease. Consult doctor immediately.")
    else:
        st.success("✅ No significant risk of heart disease is predicted. Maintain a healthy lifestyle and consult professionals for proper checkups.")

        