import streamlit as st
import joblib
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from feature_extraction import extract_features

# Load model and scaler
model = joblib.load("models/parkinson_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# App Title
st.title("🧠 Parkinson’s Disease Detection using Voice Analysis")
st.write("Upload your **voice sample** to predict Parkinson’s disease presence.")

# File uploader
uploaded_file = st.file_uploader("🎵 Upload a voice file (wav/mp3/flac)", type=["wav", "mp3", "flac"])

if uploaded_file is not None:
    # Load audio
    y, sr = librosa.load(uploaded_file, sr=None)

    # 🎧 Play back the uploaded audio
    st.subheader("🎧 Listen to Uploaded Voice Sample")
    st.audio(uploaded_file, format="audio/wav")

    # Display waveform and spectrogram
    st.subheader("📊 Audio Visualization")

    fig, ax = plt.subplots(2, 1, figsize=(10, 5))
    librosa.display.waveshow(y, sr=sr, ax=ax[0])
    ax[0].set_title("Waveform")
    S = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    img = librosa.display.specshow(S, sr=sr, x_axis="time", y_axis="log", ax=ax[1])
    ax[1].set_title("Spectrogram")
    fig.colorbar(img, ax=ax[1], format="%+2.0f dB")
    st.pyplot(fig)

    # Feature extraction and prediction
    with st.spinner("🔍 Extracting features and predicting... Please wait."):
        try:
            features = extract_features(y, sr)
            X = scaler.transform([features])
            prob = model.predict_proba(X)[0][1]
            pred = model.predict(X)[0]

            st.success(f"✅ Prediction Complete! Probability: {prob:.2f}")

            if pred == 1:
                st.error(f"🚨 Parkinson’s Detected (Probability: {prob:.2f})")
            else:
                st.success(f"💚 No Parkinson’s Detected (Probability: {prob:.2f})")

        except Exception as e:
            st.error(f"Error during prediction: {e}")

# Footer
st.markdown("---")
st.markdown("Developed by **Team HackSphere 2.0** | Powered by Streamlit + Librosa + ML 🧠")
