# 1_process_training_videos.py
import os
import glob
from config import HIGHLIGHT_VIDEOS_DIR, EMOTION_TRAINING_DATA_DIR
from _functions import process_video_for_emotions

def main():
    """
    Processes all video files in the HIGHLIGHT_VIDEOS_DIR to generate
    emotion data CSVs for training.
    """
    # Find all .mp4 files in the specified directory.
    video_paths = glob.glob(os.path.join(HIGHLIGHT_VIDEOS_DIR, "*.mp4"))

    if not video_paths:
        print(f"No .mp4 videos found in {HIGHLIGHT_VIDEOS_DIR}")
        return

    print(f"Found {len(video_paths)} videos to process for training.")

    for video_path in video_paths:
        video_filename = os.path.basename(video_path)
        print(f"\n--- Processing '{video_filename}' ---")

        # Define a path for the output video with emotion overlays (optional).
        output_video_path = os.path.join(EMOTION_TRAINING_DATA_DIR, "processed_" + video_filename)

        # Process the video. The resulting CSV will be saved in EMOTION_TRAINING_DATA_DIR.
        process_video_for_emotions(
            video_path=video_path,
            output_video_path=output_video_path,
            save_dir=EMOTION_TRAINING_DATA_DIR,
            plot_results=False  # Turn off plotting for batch processing.
        )
    
    print("\n--- All training videos processed. ---")

if __name__ == "__main__":
    main()