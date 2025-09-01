# api/main.py

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import io
import os
import cv2
import numpy as np
import joblib
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

# Define FastAPI app instance
app = FastAPI()

# --- Model and Label Encoder Loading ---
# Load model and label encoder once (on cold start)
# Use a robust path to the models relative to the script location
base_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_dir, '../ml_model/random_forest_model.pkl')
le_path = os.path.join(base_dir, '../ml_model/label_encoder.pkl')

try:
    model = joblib.load(model_path)
    label_encoder = joblib.load(le_path)
except FileNotFoundError as e:
    # A cleaner way to handle missing models
    # This will prevent the app from starting if models aren't found
    raise RuntimeError(f"Model or label encoder not found: {e}. Please check your file paths.")

# --- Feature extraction functions (reused from your original code) ---
LBP_RADIUS = 2
LBP_POINTS = 8 * LBP_RADIUS
LBP_METHOD = "uniform"
GLCM_DISTANCES = [1, 2, 3]
GLCM_ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
GLCM_PROPS = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
COLOR_BINS = 8

def extract_color_histogram(image_rgb):
    hist = cv2.calcHist([image_rgb], [0, 1, 2], None, (COLOR_BINS, COLOR_BINS, COLOR_BINS), [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def extract_lbp_features(gray_image):
    lbp = local_binary_pattern(gray_image, LBP_POINTS, LBP_RADIUS, LBP_METHOD)
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, LBP_POINTS + 3), range=(0, LBP_POINTS + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist

def extract_haralick_features(gray_image):
    glcm = graycomatrix(gray_image, distances=GLCM_DISTANCES, angles=GLCM_ANGLES, symmetric=True, normed=True)
    features = [graycoprops(glcm, prop).flatten() for prop in GLCM_PROPS]
    return np.hstack(features)

def extract_features_from_bytes(image_bytes: bytes):
    image_np = np.frombuffer(image_bytes, np.uint8)
    image_bgr = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("Image decoding failed. The uploaded file may not be a valid image.")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    gray_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    color_features = extract_color_histogram(image_rgb)
    lbp_features = extract_lbp_features(gray_image)
    haralick_features = extract_haralick_features(gray_image)
    return np.hstack([color_features, lbp_features, haralick_features])

# --- FastAPI Endpoints ---

@app.get("/")
async def read_root():
    """Simple health check endpoint to confirm the service is running."""
    return {"message": "Gemstone ML service is running and ready for predictions."}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accepts an image file and returns the predicted gemstone name.
    
    - **file**: The image file to be classified.
    """
    # 1. Read the uploaded file
    try:
        image_bytes = await file.read()
    except Exception:
        raise HTTPException(status_code=400, detail="Failed to read uploaded file.")

    # 2. Extract features from the image bytes
    try:
        features = extract_features_from_bytes(image_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception:
        raise HTTPException(status_code=500, detail="An error occurred during feature extraction.")
    
    # 3. Make a prediction using the loaded model
    try:
        features_reshaped = features.reshape(1, -1)
        pred = model.predict(features_reshaped)[0]
        gemstone_name = label_encoder.inverse_transform([pred])[0]
    except Exception:
        raise HTTPException(status_code=500, detail="An error occurred during prediction.")

    # 4. Return the prediction as a JSON response
    return JSONResponse(content={
        "gemstoneName": gemstone_name,
        "message": "Prediction successful"
    })
