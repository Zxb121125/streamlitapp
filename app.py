import streamlit as st
import joblib
import pandas as pd

model = joblib.load("model1.pkl")
col = ['Malignant Cancer', 'Sepsis', 'AKI 7Day', 'InvasiveVent', 'LAR',
       'Age', 'SBP', 'DBP', 'MBP', 'Temperature', 'Hemoglobin', 'WBC',
       'Aniongap', 'Bicarbonate', 'BUN', 'Creatinine', 'INR', 'ALP',
       'GCS', 'Total Bilirubin', 'Chloride', 'APTT', 'Sodium',
       'Weight Admit']
x1 = "Malignant Cancer	Sepsis	AKI 7Day	InvasiveVent	LAR	Age	SBP	DBP	MBP	Temperature	Hemoglobin	WBC	Aniongap	Bicarbonate	BUN	Creatinine	INR	ALP	GCS	Total Bilirubin	Chloride	APTT	Sodium	Weight Admit".split("\t")
x2 = "1	0	1	0	0.38	68	120	73	84	36.92	11.5	13.4	15	23	12	0.5	1.4	265	15	1.9	101	71.2	135	68.6".split("\t")
x2 = [eval(i) for i in x2]

originv = {i:j for i, j in zip(x1,x2)}

title = "A machine learning for predicting 30-day mortality in ICU patients with PE"

st.set_page_config(
    page_title=f"{title}",
    layout="wide"
)

st.markdown(f'''
    <h1 style="text-align: center; font-size: 26px; font-weight: bold; color: black; background: transparent; border-radius: 0rem; margin-bottom: 15px; border-bottom: 1px solid black;">
    {title}
    </h1>''', unsafe_allow_html=True)
    
BOOL = {"Yes":1, "No":0}

data = {}

with st.form("form"):
    c = st.columns(6)
    data["Age"] = c[0].number_input("Age (years)", min_value=0, max_value=120, value=originv["Age"])
    data["Weight Admit"] = c[1].number_input("Weight Admit (kg)", min_value=0.00, step=0.01, value=originv["Weight Admit"]-0.01+0.01)
    data["Malignant Cancer"] = BOOL[c[2].selectbox("Malignant Cancer", BOOL, index=originv["Malignant Cancer"])]
    data["Sepsis"] = BOOL[c[3].selectbox("Sepsis", BOOL, index=originv["Sepsis"])]
    data["AKI 7Day"] = BOOL[c[4].selectbox("AKI 7 Day", BOOL, index=originv["AKI 7Day"])]
    data["SBP"] = c[5].number_input("SBP (mmHg)", min_value=0.00, step=0.01, value=originv["SBP"]-0.01+0.01)
    data["DBP"] = c[0].number_input("DBP (mmHg)", min_value=0.00, step=0.01, value=originv["DBP"]-0.01+0.01)
    data["MBP"] = c[1].number_input("MBP (mmHg)", min_value=0.00, step=0.01, value=originv["MBP"]-0.01+0.01)
    data["Temperature"] = c[2].number_input("Temperature (℃)", min_value=0.00, step=0.01, value=originv["Temperature"]-0.01+0.01)
    data["LAR"] = c[3].number_input("LAR", min_value=0.00, step=0.01, value=originv["LAR"]-0.01+0.01)
    data["Hemoglobin"] = c[4].number_input("Hemoglobin (g/dL)", min_value=0.00, step=0.01, value=originv["Hemoglobin"]-0.01+0.01)
    data["WBC"] = c[5].number_input("WBC ($10^9$/L)", min_value=0.00, step=0.01, value=originv["WBC"]-0.01+0.01)
    data["Aniongap"] = c[0].number_input("Aniongap (mmol/L)", min_value=0.00, step=0.01, value=originv["Aniongap"]-0.01+0.01)
    data["Bicarbonate"] = c[1].number_input("Bicarbonate (mmol/L)", min_value=0.00, step=0.01, value=originv["Bicarbonate"]-0.01+0.01)
    data["Creatinine"] = c[2].number_input("Creatinine (mg/dL)", min_value=0.00, step=0.01, value=originv["Creatinine"]-0.01+0.01)
    data["BUN"] = c[3].number_input("BUN (mg/dL)", min_value=0.00, step=0.01, value=originv["BUN"]-0.01+0.01)
    data["Chloride"] = c[4].number_input("Chloride (mmol/L)", min_value=0.00, step=0.01, value=originv["Chloride"]-0.01+0.01)
    data["Sodium"] = c[5].number_input("Sodium (mmol/L)", min_value=0.00, step=0.01, value=originv["Sodium"]-0.01+0.01)
    data["INR"] = c[0].number_input("INR", min_value=0.00, step=0.01, value=originv["INR"]-0.01+0.01)
    data["APTT"] = c[1].number_input("APTT (s)", min_value=0.00, step=0.01, value=originv["APTT"]-0.01+0.01)
    data["ALP"] = c[2].number_input("ALP (U/L)", min_value=0.00, step=0.01, value=originv["ALP"]-0.01+0.01)
    data["Total Bilirubin"] = c[3].number_input("Total Bilirubin(mg/dL)", min_value=0.00, step=0.01, value=originv["Total Bilirubin"]-0.01+0.01)
    data["GCS"] = c[4].number_input("GCS Score", min_value=0.00, step=0.01, value=originv["GCS"]-0.01+0.01)
    data["InvasiveVent"] = BOOL[c[5].selectbox("Invasive Vent", BOOL, index=originv["InvasiveVent"])]
   
    c1 = st.columns(3)
    bt = c1[1].form_submit_button("Start predict", use_container_width=True, type="primary")

d = pd.DataFrame([data])
d = d[col]

if bt:
    res = round(float(model.predict_proba(d).flatten()[1])*100, 2)
    
    st.markdown(f'''
    <div style="text-align: center; font-size: 26px; color: black; margin-bottom: 5px;">
    Prediction Probability: {res}%
    </div>''', unsafe_allow_html=True)
else:
    st.markdown(f'''
    <div style="text-align: center; font-size: 26px; color: black; margin-bottom: 5px;">
    Click 'Start predict' button to start predict!
    </div>''', unsafe_allow_html=True)

st.markdown(f'''
    <hr>
    <div style="text-align: center; font-size: 20px; color: black; margin-bottom: 12px;">
    <div style="background: transparent; color: black; width: 200px; margin-left: calc(50% - 100px);">Introduction</div>
    <div style="border: 0.5px solid gray; text-align: left; font-size: 18px; padding: 12px; border-radius: 6px;">
        This web-based calculator was developed using the LASSO algorithm, achieving an AUC of 0.833 (95% CI: 0.791-0.871) and a Brier score of 0.143. Cases with predicted probabilities of 23.8% or higher are classified as being at high risk for 30-day mortality.
    </div>
    </div>''', unsafe_allow_html=True)
    
st.markdown(f'''
    <div style="text-align: center; font-size: 20px; color: black; margin-bottom: 5px;">
    <div style="background: transparent; color: black; width: 200px; margin-left: calc(50% - 100px);">Tip</div>
    <div style="border: 0.5px solid gray; color: black; text-align: left; font-size: 18px; padding: 12px; border-radius: 6px;">
        Vital signs are averaged over the first 24 hours post-lCU admission, laboratory parameters are recorded at peak severity during the ICU stay, and LAR is the initial lactate (mmol/L) to albumin (g/dL) ratio.
    </div>
    </div>''', unsafe_allow_html=True)
