# _functions.py
import os
import glob
import pickle
import joblib
import cv2
import yt_dlp
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta
from collections import Counter
from pydub import AudioSegment
from hsemotion.facial_emotions import HSEmotionRecognizer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from chat_downloader import ChatDownloader

# Import helper functions from utils.py
from utils import get_face_bbox, interpolate_nan, logits_to_probability

# --- Global Initializations ---
emotion_recognizer = HSEmotionRecognizer(model_name='enet_b0_8_best_vgaf')

# --- Main Feature Functions ---

def process_video_for_emotions(video_path, output_video_path, fps_target=5, plot_results=True, save_csv=True, save_dir="."):
    """
    Processes a video file to extract and analyze facial emotions,
    saving the results as a CSV file with 30-second averages.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Couldn't open the video file {video_path}.")
        return None

    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    if 29.95 <= actual_fps <= 30.05:
        actual_fps = 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps_target, (width, height))

    emotion_names = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']
    emotion_scores = {emotion: [] for emotion in emotion_names}

    frame_idx = 0
    frame_skip_interval = int(actual_fps / fps_target) if actual_fps > fps_target else 1

    print(f"Processing video: {os.path.basename(video_path)}...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip_interval == 0:
            bbox = get_face_bbox(frame)
            current_emotion_scores = {}
            if bbox:
                x, y, w, h = bbox
                face_roi = frame[y:y+h, x:x+w]
                if face_roi.size > 0:
                    emotion, scores = emotion_recognizer.predict_emotions(face_roi)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
                    cv2.putText(frame, f"Emotion: {emotion}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                    for i, name in enumerate(emotion_names):
                        current_emotion_scores[name] = scores[i]
                else:
                    for name in emotion_names: current_emotion_scores[name] = np.nan
            else:
                for name in emotion_names: current_emotion_scores[name] = np.nan

            for name in emotion_names:
                emotion_scores[name].append(current_emotion_scores.get(name))

            out.write(frame)

        frame_idx += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    for name in emotion_names:
        emotion_scores[name] = interpolate_nan(emotion_scores[name])
        emotion_scores[name] = logits_to_probability(np.array(emotion_scores[name]))

    interval_sec = 30
    frames_per_interval = interval_sec * fps_target
    num_intervals = len(emotion_scores[emotion_names[0]]) // frames_per_interval
    average_df = pd.DataFrame(columns=['Start_Time', 'End_Time'] + emotion_names)

    for i in range(num_intervals):
        start_frame = i * frames_per_interval
        end_frame = start_frame + frames_per_interval
        row_data = {
            'Start_Time': str(timedelta(seconds=int(start_frame / fps_target))),
            'End_Time': str(timedelta(seconds=int(end_frame / fps_target)))
        }
        for name in emotion_names:
            row_data[name] = np.mean(emotion_scores[name][start_frame:end_frame])
        average_df = pd.concat([average_df, pd.DataFrame([row_data])], ignore_index=True)

    if save_csv:
        os.makedirs(save_dir, exist_ok=True)
        csv_filename = os.path.basename(output_video_path).replace('.mp4', '_30s_avg_emotions.csv')
        output_csv_path = os.path.join(save_dir, csv_filename)
        average_df.to_csv(output_csv_path, index_label='Interval')
        print(f"Emotion averages saved to {output_csv_path}")

    if plot_results:
        average_df.drop(columns=['Start_Time', 'End_Time']).plot(kind='bar', figsize=(15, 7))
        plt.title(f'30-second Average Emotions for {os.path.basename(video_path)}')
        plt.xlabel('Time Intervals (30s each)')
        plt.ylabel('Average Emotion Probability')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return average_df


def train_emotion_models(train_data_folder, model_folder):
    """
    Trains KMeans clustering models for each emotion based on CSV files
    in the training data folder and saves the models.
    """
    train_data_files = glob.glob(os.path.join(train_data_folder, "*.csv"))
    if not train_data_files:
        print(f"No training CSV files found in {train_data_folder}. Aborting.")
        return

    all_train_data = [pd.read_csv(file, index_col='Interval').drop(columns=['Start_Time', 'End_Time']) for file in train_data_files]
    train_df = pd.concat(all_train_data, axis=0)
    emotions = train_df.columns
    scaler = StandardScaler()

    os.makedirs(model_folder, exist_ok=True)

    for emotion in emotions:
        emotion_data = train_df[[emotion]].dropna().values
        if emotion_data.shape[0] < 11: # Need enough samples for clustering range
            print(f"Not enough data for '{emotion}' to train a model. Skipping.")
            continue

        train_emotion_data_scaled = scaler.fit_transform(emotion_data)
        range_n_clusters = list(range(2, min(11, train_emotion_data_scaled.shape[0])))
        silhouette_avg = []
        for n_clusters in range_n_clusters:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(train_emotion_data_scaled)
            silhouette_avg.append(silhouette_score(train_emotion_data_scaled, cluster_labels))

        if not silhouette_avg:
            print(f"Could not determine optimal clusters for {emotion}. Skipping.")
            continue
            
        optimal_n_clusters = range_n_clusters[np.argmax(silhouette_avg)]
        print(f"Optimal number of clusters for {emotion}: {optimal_n_clusters}")

        kmeans = KMeans(n_clusters=optimal_n_clusters, random_state=42, n_init=10)
        kmeans.fit(train_emotion_data_scaled)

        model_path = os.path.join(model_folder, f"{emotion}_kmeans_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(kmeans, f)
        print(f"Saved model for {emotion} to {model_path}")


def label_important_scenes(training_data_folder, test_csv_path, output_labeled_path):
    """
    Labels scenes in the test CSV as 'Important' or 'Normal' based on
    dominant cluster analysis and top 10% scores for Fear/Surprise.
    """
    test_df = pd.read_csv(test_csv_path, index_col='Interval')
    emotions = [col for col in test_df.columns if col not in ['Start_Time', 'End_Time']]
    scaler = StandardScaler()

    train_files = glob.glob(os.path.join(training_data_folder, "*.csv"))
    if not train_files:
        print(f"No training data found in {training_data_folder}. Cannot perform dominant cluster analysis.")
        return test_df
    
    train_df = pd.concat([pd.read_csv(f) for f in train_files], ignore_index=True)

    for emotion in emotions:
        if emotion not in train_df.columns: continue
        
        model_path = os.path.join(os.path.dirname(training_data_folder), 'kmeans_models', f"{emotion}_kmeans_model.pkl")
        if not os.path.exists(model_path):
            print(f"Model for {emotion} not found. Skipping dominant cluster labeling.")
            continue

        kmeans = joblib.load(model_path)
        train_emotion_data = train_df[[emotion]].dropna().values
        test_emotion_data = test_df[[emotion]].dropna().values

        if len(train_emotion_data) == 0 or len(test_emotion_data) == 0: continue

        scaler.fit(train_emotion_data)
        train_data_scaled = scaler.transform(train_emotion_data)
        test_data_scaled = scaler.transform(test_emotion_data)

        # Determine dominant cluster from training data
        train_labels = kmeans.predict(train_data_scaled)
        dominant_cluster = Counter(train_labels).most_common(1)[0][0]

        test_labels = kmeans.predict(test_data_scaled)
        test_df[f'{emotion}_Cluster_Label'] = test_labels
        test_df[f'{emotion}_Important_Label'] = np.where(test_labels == dominant_cluster, 'Important', 'Normal')

    for emotion in ['Fear', 'Surprise']:
        if emotion in test_df.columns:
            threshold = test_df[emotion].quantile(0.90)
            test_df[f'{emotion}_Top10%_Important_Label'] = np.where(test_df[emotion] >= threshold, 'Important', 'Normal')

    test_df.to_csv(output_labeled_path)
    print(f"Labeled data saved to {output_labeled_path}")
    return test_df


def analyze_youtube_comments(video_url, top_n):
    """
    Analyzes YouTube comment activity over time.
    Updated to provide more robust error handling and progress feedback.
    """
    print(f"Analyzing comments for: {video_url}")
    try:
        # get_chat may timeout on long videos, so we iterate carefully
        chat = ChatDownloader().get_chat(video_url, message_groups=['messages'])
        
        timestamps = []
        try:
            for message in chat:
                timestamps.append(int(message["time_in_seconds"]))
                # Optional: Print progress every 1000 comments to ensure it's running
                if len(timestamps) % 5000 == 0:
                    print(f"  ...processed {len(timestamps)} comments so far (Current time: {str(timedelta(seconds=timestamps[-1]))})")
        except Exception as e:
            print(f"Warning: Chat download stopped unexpectedly: {e}")
            # Do not return None here, use whatever data we gathered so far
        
        if not timestamps:
            print("No comments found.")
            return None

        print(f"Total comments collected: {len(timestamps)}")
        print(f"Last comment timestamp: {str(timedelta(seconds=timestamps[-1]))}")

        # Aggregate comments into 30-second buckets
        bucket_size_sec = 30
        buckets = [t // bucket_size_sec for t in timestamps]
        counter = Counter(buckets)
        
        if not counter:
            print("Could not count any comments in buckets.")
            return None

        # Create a DataFrame
        df = pd.DataFrame(counter.items(), columns=['Interval_Index', 'Comment_Count'])
        
        # Ensure top_n does not exceed available rows
        current_top_n = min(top_n, len(df))
        
        # Get the top N intervals
        top_intervals = df.nlargest(current_top_n, 'Comment_Count')

        # Add Start_Time and End_Time
        top_intervals['Start_Time'] = top_intervals['Interval_Index'].apply(lambda idx: str(timedelta(seconds=idx * bucket_size_sec)))
        top_intervals['End_Time'] = top_intervals['Interval_Index'].apply(lambda idx: str(timedelta(seconds=(idx + 1) * bucket_size_sec)))

        return top_intervals.sort_values(by='Interval_Index').reset_index(drop=True)

    except Exception as e:
        print(f"An critical error occurred during comment analysis: {e}")
        traceback.print_exc()
        return None


def analyze_audio_from_local(video_path, top_n):
    """
    Analyzes audio loudness from the LOCAL video file instead of downloading from YouTube.
    This ensures 100% synchronization with the video analysis and avoids download errors.
    """
    print(f"Analyzing audio from local file: {os.path.basename(video_path)}")
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return None

    try:
        # Load audio directly from the video file using pydub
        # pydub uses ffmpeg to extract the audio stream
        audio = AudioSegment.from_file(video_path)
        
        duration_sec = len(audio) / 1000
        print(f"Audio loaded. Duration: {str(timedelta(seconds=int(duration_sec)))}")

        step_ms = 30 * 1000 # Set interval to 30 seconds
        loudness_per_interval = []
        
        # Analyze in chunks
        for i in range(0, len(audio), step_ms):
            segment = audio[i:i+step_ms]
            if segment.duration_seconds > 0:
                loudness_per_interval.append(segment.dBFS)

        if not loudness_per_interval:
            print("Could not analyze audio loudness.")
            return None
            
        # Create a DataFrame
        df = pd.DataFrame({
            'Interval_Index': range(len(loudness_per_interval)),
            'Loudness_dBFS': loudness_per_interval
        })

        # Ensure top_n does not exceed available rows
        current_top_n = min(top_n, len(df))

        # Get the top N intervals
        top_intervals = df.nlargest(current_top_n, 'Loudness_dBFS')

        # Add Start_Time and End_Time
        interval_duration_sec = step_ms / 1000
        top_intervals['Start_Time'] = top_intervals['Interval_Index'].apply(lambda idx: str(timedelta(seconds=idx * interval_duration_sec)))
        top_intervals['End_Time'] = top_intervals['Interval_Index'].apply(lambda idx: str(timedelta(seconds=(idx + 1) * interval_duration_sec)))

        return top_intervals.sort_values(by='Interval_Index').reset_index(drop=True)

    except Exception as e:
        print(f"An error occurred during local audio analysis: {e}")
        import traceback
        traceback.print_exc()
        return None


def select_and_sort_highlight_scenes(labeled_data_path, highlight_duration_minutes=10):
    """
    Selects and sorts the most important scenes based on the labeled CSV data
    to generate a final list of highlight timestamps.
    """
    if not os.path.exists(labeled_data_path):
        print(f"Error: Labeled data file not found at {labeled_data_path}.")
        return None

    data = pd.read_csv(labeled_data_path, index_col='Interval')
    important_labels = [col for col in data.columns if col.endswith('_Important_Label')]
    
    if 'Start_Time' not in data.columns or 'End_Time' not in data.columns:
        print("Error: 'Start_Time' or 'End_Time' columns are missing.")
        return None

    data["Importance_Score"] = data[important_labels].apply(lambda row: sum(row == "Important"), axis=1)
    
    # 10 minutes * 2 intervals/minute = 20 intervals to select
    num_intervals_to_select = int(highlight_duration_minutes * 2)
    num_intervals_to_select = min(num_intervals_to_select, len(data))

    highlight_scenes = data.nlargest(num_intervals_to_select, "Importance_Score")
    selected_scenes = highlight_scenes[["Start_Time", "End_Time", "Importance_Score"]].copy()

    # Sort by time
    selected_scenes["Start_Time_Sec"] = pd.to_timedelta(selected_scenes["Start_Time"]).dt.total_seconds()
    selected_scenes_sorted = selected_scenes.sort_values(by="Start_Time_Sec").drop(columns=["Start_Time_Sec"])

    return selected_scenes_sorted

# --- Add the new experiment functions below ---

def select_scenes_by_emotion_intensity(emotion_csv_path, top_n):
    """
    Selects top N scenes based on the highest single emotion score in each interval,
    regardless of which emotion it is.
    """
    if not os.path.exists(emotion_csv_path):
        print(f"Error: Emotion CSV not found at {emotion_csv_path}.")
        return None

    df = pd.read_csv(emotion_csv_path, index_col='Interval')
    emotion_columns = ['Anger', 'Contempt', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']
    
    # Make sure all required emotion columns exist
    valid_emotion_columns = [col for col in emotion_columns if col in df.columns]
    if not valid_emotion_columns:
        print("Error: No emotion columns found in the CSV.")
        return None

    # For each row, find the maximum emotion probability and the name of that emotion
    df['Max_Emotion_Score'] = df[valid_emotion_columns].max(axis=1)
    df['Dominant_Emotion'] = df[valid_emotion_columns].idxmax(axis=1)

    # Select the top N scenes based on the new 'Max_Emotion_Score'
    top_scenes = df.nlargest(top_n, 'Max_Emotion_Score')
    
    # Sort the final list by time for easier review
    top_scenes_sorted = top_scenes.sort_index()

    return top_scenes_sorted[['Start_Time', 'End_Time', 'Dominant_Emotion', 'Max_Emotion_Score']]

import traceback
def select_scenes_by_comments(video_url, top_n):
    """
    Selects top N 30-second intervals based on comment count.
    """
    print(f"Analyzing comments for top {top_n} scenes: {video_url}")
    try:
        chat = ChatDownloader().get_chat(video_url)
        timestamps = [int(message["time_in_seconds"]) for message in chat]
        if not timestamps:
            print("No comments found.")
            return None

        # Aggregate comments into 30-second buckets
        bucket_size_sec = 30
        buckets = [t // bucket_size_sec for t in timestamps]
        counter = Counter(buckets)
        
        if not counter:
            print("Could not count any comments in buckets.")
            return None

        # Create a DataFrame
        df = pd.DataFrame(counter.items(), columns=['Interval_Index', 'Comment_Count'])
        
        # Get the top N intervals
        top_intervals = df.nlargest(top_n, 'Comment_Count')

        # Add Start_Time and End_Time for clarity
        top_intervals['Start_Time'] = top_intervals['Interval_Index'].apply(lambda idx: str(timedelta(seconds=idx * bucket_size_sec)))
        top_intervals['End_Time'] = top_intervals['Interval_Index'].apply(lambda idx: str(timedelta(seconds=(idx + 1) * bucket_size_sec)))

        # Sort by time
        return top_intervals.sort_values(by='Interval_Index').reset_index(drop=True)

    except Exception as e:
        print(f"An error occurred during comment analysis: {e}")
        print(f"An error occurred during comment analysis: {e}")
        print("\n--- FULL TRACEBACK ---")
        traceback.print_exc()  # エラーの詳細な履歴（トレースバック）をすべて表示
        print("--- END TRACEBACK ---")
        return None


def select_scenes_by_loudness(video_url, top_n, temp_dir="/tmp"):
    """
    Selects top N 30-second intervals based on audio loudness (dBFS).
    """
    print(f"Analyzing audio for top {top_n} scenes: {video_url}")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(temp_dir, "temp_audio.%(ext)s"),
        "postprocessors": [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3'}],
        "quiet": True,
    }
    audio_file_path = os.path.join(temp_dir, "temp_audio.mp3")

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])

        if not os.path.exists(audio_file_path):
            print("Audio download failed.")
            return None

        audio = AudioSegment.from_file(audio_file_path)
        step_ms = 30 * 1000
        loudness_per_interval = []
        for i in range(0, len(audio), step_ms):
            segment = audio[i:i+step_ms]
            if segment.duration_seconds > 0:
                loudness_per_interval.append(segment.dBFS)

        if not loudness_per_interval:
            print("Could not analyze audio loudness.")
            return None
            
        # Create a DataFrame
        df = pd.DataFrame({
            'Interval_Index': range(len(loudness_per_interval)),
            'Loudness_dBFS': loudness_per_interval
        })

        # Get the top N intervals
        top_intervals = df.nlargest(top_n, 'Loudness_dBFS')

        # Add Start_Time and End_Time for clarity
        interval_duration_sec = step_ms / 1000
        top_intervals['Start_Time'] = top_intervals['Interval_Index'].apply(lambda idx: str(timedelta(seconds=idx * interval_duration_sec)))
        top_intervals['End_Time'] = top_intervals['Interval_Index'].apply(lambda idx: str(timedelta(seconds=(idx + 1) * interval_duration_sec)))

        return top_intervals.sort_values(by='Interval_Index').reset_index(drop=True)

    except Exception as e:
        print(f"An error occurred during audio analysis: {e}")
        return None
    finally:
        if os.path.exists(audio_file_path):
            os.remove(audio_file_path)