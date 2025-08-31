import joblib
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
import io

# --- Initialize the FastAPI App ---
app = FastAPI(
    title="GEMX Prediction API",
    description="A high-speed API for gemstone classification using advanced feature extraction.",
    version="1.0.0"
)

# --- 1. Load Models and Encoders ONCE on Startup ---
# This is the key to high performance. The models are kept in memory.
try:
    model = joblib.load('./ml_model/random_forest_model.pkl')
    label_encoder = joblib.load('./ml_model/label_encoder.pkl')
    print("✅ Model and Label Encoder loaded successfully.")
except Exception as e:
    model = None
    label_encoder = None
    print(f"❌ ERROR: Could not load model or encoder. The API will not work. Error: {e}")

# --- 2. Feature Extraction Logic (Copied directly from your predict.py) ---
# Parameters MUST be identical to those used during model training.
LBP_RADIUS = 2
LBP_POINTS = 8 * LBP_RADIUS
LBP_METHOD = "uniform"
GLCM_DISTANCES = [1, 2, 3]
GLCM_ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
GLCM_PROPS = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM"]
COLOR_BINS = 8

def extract_color_histogram(image_rgb):
    """Calculates a 3D color histogram from an RGB image."""
    hist = cv2.calcHist([image_rgb], [0, 1, 2], None, (COLOR_BINS, COLOR_BINS, COLOR_BINS), [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def extract_lbp_features(gray_image):
    """Extracts LBP texture features from a grayscale image."""
    lbp = local_binary_pattern(gray_image, LBP_POINTS, LBP_RADIUS, LBP_METHOD)
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, LBP_POINTS + 3), range=(0, LBP_POINTS + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist

def extract_haralick_features(gray_image):
    """Extracts Haralick texture features from a grayscale image."""
    glcm = graycomatrix(gray_image, distances=GLCM_DISTANCES, angles=GLCM_ANGLES, symmetric=True, normed=True)
    features = [graycoprops(glcm, prop).flatten() for prop in GLCM_PROPS]
    return np.hstack(features)

# --- 3. Main Preprocessing Pipeline for API ---
# This function takes the raw image bytes from the web request and processes them.
def extract_features_from_bytes(image_bytes: bytes) -> np.ndarray:
    """Loads an image from bytes and extracts the full feature vector."""
    try:
        # Convert the byte stream to a NumPy array
        image_np = np.frombuffer(image_bytes, np.uint8)
        # Decode the NumPy array into a BGR image using OpenCV
        image_bgr = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        if image_bgr is None:
            raise ValueError("Image decoding failed. The file may be corrupt or in an unsupported format.")

        # Convert to RGB for color features (to match your training process)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # Convert to grayscale for texture features
        gray_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        # Extract all feature types using the functions above
        color_features = extract_color_histogram(image_rgb)
        lbp_features = extract_lbp_features(gray_image)
        haralick_features = extract_haralick_features(gray_image)

        # Combine all features into a single vector
        return np.hstack([color_features, lbp_features, haralick_features])
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process image. Error: {e}")

# --- 4. The Prediction Endpoint ---
@app.post("/predict", summary="Predict Gemstone from Image")
async def predict(image: UploadFile = File(..., description="An image file of the gemstone.")):
    """
    Receives an image, extracts a complex feature vector, and returns the predicted gemstone name.
    """
    if not model or not label_encoder:
        raise HTTPException(status_code=503, detail="Model or encoder is not available. Please check server logs.")

    # Read the image content from the HTTP request
    image_bytes = await image.read()
    
    # Extract the full feature vector from the image bytes
    features = extract_features_from_bytes(image_bytes)
    
    # Make the prediction
    try:
        # Reshape features for a single prediction
        features_reshaped = features.reshape(1, -1)
        # Get the numeric prediction from the model
        numeric_prediction = model.predict(features_reshaped)
        # Convert the numeric prediction back to the original text label
        text_prediction = label_encoder.inverse_transform(numeric_prediction)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction failed. Error: {e}")

    return {
        "gemstoneName": text_prediction[0],
        "message": "Prediction successful"
    }

# --- 5. Root Endpoint for Health Check ---
@app.get("/", summary="Health Check")
def read_root():
    """A simple endpoint to confirm the API is running."""
    return {"status": "GEMX Prediction API is running!"}

