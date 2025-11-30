# config.py

# --- Directory Paths ---
HIGHLIGHT_VIDEOS_DIR = "/home/ubuntu/Videos/highlights/" # 学習用のハイライト動画が入っているフォルダ
ORIGINAL_VIDEO_PATH = "/home/ubuntu/Videos/sample15.original.mp4" # 分析したいオリジナル動画のパス

EMOTION_TRAINING_DATA_DIR = "/home/ubuntu/Documents/emotion_data_folder/" # 学習用CSVの保存先
KMEANS_MODEL_DIR = "/home/ubuntu/Documents/kmeans_models/" # 学習済みモデルの保存先
ANALYSIS_RESULTS_DIR = "/home/ubuntu/Documents/analysis_results/" # オリジナル動画の分析結果の保存先

# --- Model Paths ---
SHAPE_PREDICTOR_PATH = "/home/ubuntu/models/shape_predictor_68_face_landmarks.dat"

# --- YouTube URL ---
YOUTUBE_URL = "https://www.youtube.com/live/L7WGAXzkSQI?si=PF0_o66ptKnMn_ih"