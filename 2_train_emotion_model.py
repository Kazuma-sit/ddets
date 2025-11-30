# 2_train_emotion_model.py
from config import EMOTION_TRAINING_DATA_DIR, KMEANS_MODEL_DIR
from functions import train_emotion_models

def main():
    """
    Trains KMeans models based on the aggregated emotion data from CSVs
    and saves the trained models to the specified directory.
    """
    print("--- Starting emotion model training... ---")
    train_emotion_models(EMOTION_TRAINING_DATA_DIR, KMEANS_MODEL_DIR)
    print("\n--- Model training complete. Models are saved in '{KMEANS_MODEL_DIR}'. ---")

if __name__ == "__main__":
    main()