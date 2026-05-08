import streamlit as st
import joblib
import pandas as pd

model = joblib.load('model.joblib')

st.title('Titanic Survived Prediction')
st.write('Masukkan data penumpang untuk melihat prediksi keselamatan')

col1, col2 = st.columns(2)
with col1:
    pclass = st.selectbox("Kelas Tiket (Pclass)", [1, 2, 3])
    sex = st.selectbox("Jenis Kelamin", ["male", "female"])
    sibsp = st.number_input("Jumlah Saudara/Pasangan (SibSp)", min_value=0, step=1)

with col2:
    parch = st.number_input("Jumlah Orang Tua/Anak (Parch)", min_value=0, step=1)
    fare = st.number_input("Tarif Tiket (Fare)", min_value=0.0)
    embarked = st.selectbox("Pelabuhan Keberangkatan", ["S", "C", "Q"])

# tombol prediksi
if st.button('PREDIKSI'):
    # Data harus sama dengan training
    data = pd.DataFrame({
    'Pclass': [pclass],
    'Sex': [sex],
    'SibSp': [sibsp],
    'Parch': [parch],
    'Fare': [fare],
    'Embarked': [embarked]
    })

    prediction = model.predict(data)[0]

    proba = model.predict_proba(data)[0][1]

    if prediction == 1:
        st.success("✅ Penumpang diprediksi SELAMAT")
    else:
        st.error("❌ Penumpang diprediksi TIDAK SELAMAT")

    st.info(f"Probabilitas selamat: {proba:.2%}")

