import streamlit as st
import pandas as pd
import joblib

model = joblib.load("models/random_forest_patient_safety.pkl")
encoder = joblib.load("models/label_encoder.pkl")

st.set_page_config(page_title="Transparent AI for Patient Safety")

st.title("🩺 Transparent AI for Patient Safety")
st.subheader("Drug Safety Risk Prediction System")

st.write("Enter drug details to predict **High Risk / Low Risk** with explanation.")


drug_name = st.text_input("Drug Name", "Isotretinoin")
medical_condition = st.text_input("Medical Condition", "Acne")
drug_class = st.text_input("Drug Class", "Retinoids")

rx_otc = st.selectbox("Prescription Type", ["Rx", "OTC"])
pregnancy = st.selectbox("Pregnancy Category", ["A", "B", "C", "D", "X", "N"])

rating = st.slider("Drug Rating", 0.0, 10.0, 5.0)
no_of_reviews = st.number_input("Number of Reviews", min_value=0, value=100)


if st.button("Predict Safety Risk"):
    input_df = pd.DataFrame([{
        "drug_name": drug_name,
        "medical_condition": medical_condition,
        "drug_classes": drug_class,
        "rx_otc": rx_otc,
        "pregnancy_category": pregnancy,
        "rating": rating,
        "no_of_reviews": no_of_reviews
    }])

    cat_cols = [
        "drug_name",
        "medical_condition",
        "drug_classes",
        "rx_otc",
        "pregnancy_category"
    ]

    for col in cat_cols:
        input_df[col] = encoder.fit_transform(input_df[col].astype(str))

    prediction = model.predict(input_df)[0]

    st.markdown("---")
    if prediction == 1:
        st.error("⚠️ High Safety Risk")
        st.write("### Explanation:")
        if pregnancy in ["D", "X"]:
            st.write("- Drug is unsafe during pregnancy.")
        if rating < 6:
            st.write("- Drug has a low safety rating.")
        st.write("- Model predicts this drug may pose a safety risk.")
    else:
        st.success("✅ Low Safety Risk")
        st.write("### Explanation:")
        st.write("- Pregnancy category is safe.")
        st.write("- Drug rating is acceptable.")
        st.write("- Model predicts low safety risk.")
