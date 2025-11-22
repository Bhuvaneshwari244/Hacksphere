<<<<<<< HEAD
# Hacksphere
=======

# Parkinson's Voice Detection - HackSphere 2.0

## Contents
- `app.py` - Flask API for audio upload and prediction
- `feature_extraction.py` - audio loading + feature extraction (librosa)
- `model_train.py` - script to train a RandomForest on UCI dataset CSV
- `model/` - output directory where model/model.pkl will be saved after training
- `uploads/` - audio uploads

## How to use
1. Install dependencies:
   pip install -r requirements.txt

2. Train a model:
   - Download UCI Parkinson's dataset CSV and run:
     python model_train.py path/to/parkinsons.csv
   - This will create `model/model.pkl`.

3. Run the API:
   python app.py

4. Predict:
   POST a wav/mp3 file to `http://localhost:5000/predict` as form-data `file`.

   After deployement check the details : https://bhuvaneshwari244-hacksphere-app-ui-qfswxl.streamlit.app/

   commands to execute the code:
   1) python train_model.py
   2) streamlit run app_ui.py

## Notes
- Jitter and shimmer are approximated; for precise clinical features use specialized algorithms.
- Replace placeholder example metrics in the presentation after training & evaluation.
>>>>>>> 956af47 (Initial commit - Parkinson’s Detection project)
