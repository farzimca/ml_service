# api/predict.py

from http.server import BaseHTTPRequestHandler
import io
import os
import json
import cv2
import numpy as np
import joblib
from urllib.parse import parse_header
from email.parser import BytesParser
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

# Load model and label encoder once (on cold start)
base_dir = os.path.dirname(__file__)
model = joblib.load(os.path.join(base_dir, '../ml_model/random_forest_model.pkl'))
label_encoder = joblib.load(os.path.join(base_dir, '../ml_model/label_encoder.pkl'))

# --- Feature extraction functions (copied from your app) ---
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
        raise ValueError("Image decoding failed.")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    gray_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    color_features = extract_color_histogram(image_rgb)
    lbp_features = extract_lbp_features(gray_image)
    haralick_features = extract_haralick_features(gray_image)
    return np.hstack([color_features, lbp_features, haralick_features])


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_type = self.headers.get('Content-Type')
        if not content_type:
            self.send_error(400, 'Missing Content-Type header')
            return

        ctype, pdict = parse_header(content_type)
        if ctype != 'multipart/form-data':
            self.send_error(400, 'Content-Type must be multipart/form-data')
            return

        pdict['boundary'] = pdict['boundary'].encode('utf-8')
        content_len = int(self.headers.get('Content-Length'))
        body = self.rfile.read(content_len)

        msg = BytesParser().parsebytes(b'Content-Type: ' + content_type.encode() + b'\n\n' + body)
        file_content = None

        for part in msg.walk():
            if part.get_content_disposition() == 'form-data' and part.get_filename():
                file_content = part.get_payload(decode=True)
                break

        if not file_content:
            self.send_error(400, 'No image file uploaded.')
            return

        try:
            features = extract_features_from_bytes(file_content).reshape(1, -1)
            pred = model.predict(features)
            label = label_encoder.inverse_transform(pred)[0]

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({
                'gemstoneName': label,
                'message': 'Prediction successful'
            }).encode())

        except Exception as e:
            self.send_error(500, f"Prediction failed: {str(e)}")
