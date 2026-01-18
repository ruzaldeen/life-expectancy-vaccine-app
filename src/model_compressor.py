import joblib
import os

MODEL_DIR = os.path.join("..", "models")
MODEL_DATA_UNCOMPRESSED = "model.pkl"
MODEL_DATA_COMPRESSED = "model.pkl.gz"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_DATA_UNCOMPRESSED)

def compress_model():
    # Load your existing large model
    model = joblib.load(MODEL_PATH)

    # Save it using LZMA compression (extreme compression)
    print(f"Compressing model to {MODEL_PATH}...")
    joblib.dump(model, os.path.join(MODEL_DIR, MODEL_DATA_COMPRESSED), compress=('gzip',3))

    print("Compression complete. Able to upload 'model.pkl.gz' to GitHub.")

def remove_uncompressed_model():
    # Optionally, remove the uncompressed model file to save space
    os.remove(MODEL_PATH)   
    print(f"Removed uncompressed model file: {MODEL_PATH}")

if __name__ == "__main__":
    compress_model()
    remove_uncompressed_model()