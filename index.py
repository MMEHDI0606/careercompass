from http.server import BaseHTTPRequestHandler
import json
import joblib
import pandas as pd
import numpy as np
import requests
import os

# --- Model and Component Loading ---

# Function to download a file from a URL if it doesn't exist locally
def download_file(url, local_filename):
    # In a serverless environment, files are stored in /tmp
    local_filepath = os.path.join('/tmp', local_filename)
    if not os.path.exists(local_filepath):
        print(f"Downloading {local_filename}...")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_filepath, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    return local_filepath

# URLs for your model files stored on GitHub (using Git LFS)
# This URL is now correctly formatted for your repository.
base_url = "https://media.githubusercontent.com/media/MMEHDI0606/careercompass/main/"
model_url = base_url + "xgboost_career_model.pkl"
preprocessor_url = base_url + "preprocessor.pkl"
label_encoder_url = base_url + "label_encoder.pkl"

# Download and load the model components
try:
    model_path = download_file(model_url, "xgboost_career_model.pkl")
    preprocessor_path = download_file(preprocessor_url, "preprocessor.pkl")
    label_encoder_path = download_file(label_encoder_url, "label_encoder.pkl")

    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    label_encoder = joblib.load(label_encoder_path)
    print("Model and components loaded successfully.")
except Exception as e:
    print(f"Error loading model components: {e}")
    model, preprocessor, label_encoder = None, None, None


# --- HTTP Request Handler ---

class handler(BaseHTTPRequestHandler):
    
    def do_OPTIONS(self):
        self.send_response(200, "ok")
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS')
        self.send_header("Access-Control-Allow-Headers", "X-Requested-With, Content-type")
        self.end_headers()

    def do_POST(self):
        if not all([model, preprocessor, label_encoder]):
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Model components not loaded"}).encode())
            return

        try:
            # Get the size of the incoming data
            content_length = int(self.headers['Content-Length'])
            # Read the data
            post_data = self.rfile.read(content_length)
            # Parse the JSON data
            user_input = json.loads(post_data)

            # Convert to a DataFrame
            input_df = pd.DataFrame([user_input])

            # Preprocess the input data
            input_processed = preprocessor.transform(input_df)

            # Make a prediction
            prediction_encoded = model.predict(input_processed)

            # Decode the prediction
            predicted_career = label_encoder.inverse_transform(prediction_encoded)[0]

            # --- Prepare the response ---
            self.send_response(200)
            # Add CORS headers to allow requests from any domain (e.g., Netlify)
            self.send_header('Access-Control-Allow-Origin', '*')
            self.send_header('Content-type', 'application/json')
            self.end_headers()

            response = {
                "predicted_career": predicted_career.replace('_', ' ')
            }
            self.wfile.write(json.dumps(response).encode())

        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())
        return

