# 3_analyze_original_video.py
import os
from config import (
    ORIGINAL_VIDEO_PATH,
    EMOTION_TRAINING_DATA_DIR,
    ANALYSIS_RESULTS_DIR,
    YOUTUBE_URL
)
from _functions import (
    process_video_for_emotions,
    label_important_scenes,
    analyze_youtube_comments,
    analyze_youtube_audio
)

def main():
    """
    Runs the full analysis pipeline for a single original video:
    1. Generates emotion data if it doesn't exist.
    2. Labels scenes using pre-trained models.
    3. Analyzes associated YouTube comments and audio.
    """
    os.makedirs(ANALYSIS_RESULTS_DIR, exist_ok=True)
    video_basename = os.path.basename(ORIGINAL_VIDEO_PATH)

    # Define the path for the raw emotion CSV of the original video.
    emotion_csv_path = os.path.join(
        ANALYSIS_RESULTS_DIR,
        video_basename.replace('.mp4', '_30s_avg_emotions.csv')
    )

    # Step 1: Process the video for emotions ONLY if the CSV doesn't exist.
    if not os.path.exists(emotion_csv_path):
        print(f"Emotion CSV not found. Processing '{video_basename}' now...")
        output_video_path = os.path.join(ANALYSIS_RESULTS_DIR, "processed_" + video_basename)
        
        process_video_for_emotions(
            video_path=ORIGINAL_VIDEO_PATH,
            output_video_path=output_video_path,
            save_dir=ANALYSIS_RESULTS_DIR, # Save to the dedicated analysis folder.
            plot_results=True
        )
    else:
        print(f"Emotion CSV already exists at '{emotion_csv_path}'. Skipping video processing.")

    # Step 2: Label the scenes using the trained models.
    print("\n--- Labeling important scenes based on trained models... ---")
    labeled_csv_path = os.path.join(
        ANALYSIS_RESULTS_DIR,
        video_basename.replace('.mp4', '_labeled_scenes.csv')
    )
    
    label_important_scenes(
        training_data_folder=EMOTION_TRAINING_DATA_DIR, # Models are built from this data.
        test_csv_path=emotion_csv_path,           # This is the file to label.
        output_labeled_path=labeled_csv_path      # This is the output file.
    )

    # Step 3 (Optional): Analyze comments and audio from YouTube.
    print("\n--- Analyzing YouTube comments and audio... ---")
    analyze_youtube_comments(YOUTUBE_URL)
    analyze_youtube_audio(YOUTUBE_URL)

    print(f"\n--- Analysis complete. Labeled data is available at: '{labeled_csv_path}' ---")


if __name__ == "__main__":
    main()