# 5_run_experiment.py
import os
import pandas as pd
from config import ORIGINAL_VIDEO_PATH, ANALYSIS_RESULTS_DIR, YOUTUBE_URL
from _functions import (
    select_scenes_by_emotion_intensity,
    select_scenes_by_comments,
    analyze_audio_from_local
)

def main():
    """
    Runs an experiment to compare top scenes selected by three different criteria:
    1. Emotional Intensity (from video analysis)
    2. Comment Count (from YouTube)
    3. Audio Loudness (from YouTube)
    """
    TOP_N = 50  # 抽出するシーン数をここで一括指定

    # Set pandas to display all rows to see the full comparison
    pd.set_option('display.max_rows', None)

    print("--- Starting Highlight Scene Comparison Experiment ---")

    # --- Criterion 1: Emotional Intensity ---
    print("\n" + "="*50)
    print(f"  1. Top {TOP_N} Scenes by EMOTIONAL INTENSITY (per 30 seconds)")
    print("="*50)
    video_basename = os.path.basename(ORIGINAL_VIDEO_PATH)
    csv_filename = "processed_" + video_basename.replace('.mp4', '_30s_avg_emotions.csv')
    emotion_csv_path = os.path.join(
        ANALYSIS_RESULTS_DIR,
        csv_filename
    )
    
    if not os.path.exists(emotion_csv_path):
        print(f"Error: Emotion CSV not found at '{emotion_csv_path}'")
        print("Please run '3_analyze_original_video.py' first to generate it.")
    else:
        top_emotion_scenes = select_scenes_by_emotion_intensity(emotion_csv_path, top_n=TOP_N)
        if top_emotion_scenes is not None:
            print(top_emotion_scenes)

    # --- Criterion 2: Comment Count ---
    print("\n" + "="*50)
    print(f"  2. Top {TOP_N} Scenes by COMMENT COUNT (per 30 seconds)")
    print("="*50)
    top_comment_scenes = select_scenes_by_comments(YOUTUBE_URL, top_n=TOP_N)
    if top_comment_scenes is not None:
        # 'Interval_Index' 列名を '30s_Interval_Index' にリネームして表示を分かりやすくする
        print(top_comment_scenes.rename(columns={'Interval_Index': '30s_Interval_Index'}))


    # --- Criterion 3: Audio Loudness ---
    print("\n" + "="*50)
    print(f"  3. Top {TOP_N} Scenes by AUDIO LOUDNESS (per 30 seconds)")
    print("="*50)
    top_loudness_scenes = analyze_audio_from_local(ORIGINAL_VIDEO_PATH, top_n=TOP_N)
    if top_loudness_scenes is not None:
        # 'Interval_Index' 列名を '30s_Interval_Index' にリネームして表示を分かりやすくする
        print(top_loudness_scenes.rename(columns={'Interval_Index': '30s_Interval_Index'}))
    
    print("\n--- Experiment Finished ---")


if __name__ == "__main__":
    main()
