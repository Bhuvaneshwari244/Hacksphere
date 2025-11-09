import numpy as np
import librosa

def extract_features(y, sr):
    """
    Extract 30 MFCC features to match the trained model input shape.
    """
    # Extract 30 MFCCs (used during training)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=30).T, axis=0)
    return mfccs
