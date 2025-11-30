# 4_generate_highlights.py
import os
from config import ORIGINAL_VIDEO_PATH, ANALYSIS_RESULTS_DIR
from _functions import select_and_sort_highlight_scenes

def main():
    """
    Takes the final labeled CSV file and generates a sorted list
    of highlight scene timestamps.
    """
    video_basename = os.path.basename(ORIGINAL_VIDEO_PATH)
    labeled_csv_path = os.path.join(
        ANALYSIS_RESULTS_DIR,
        video_basename.replace('.mp4', '_labeled_scenes.csv')
    )

    if not os.path.exists(labeled_csv_path):
        print(f"Error: Labeled CSV not found at '{labeled_csv_path}'")
        print("Please run '3_analyze_original_video.py' first to generate it.")
        return

    print("--- Selecting and sorting final highlight scenes... ---")
    final_highlights_df = select_and_sort_highlight_scenes(
        labeled_data_path=labeled_csv_path,
        highlight_duration_minutes=10
    )

    if final_highlights_df is not None:
        print("\n--- Final Highlight Scenes (Sorted by Time) ---")
        print(final_highlights_df.to_string()) # Use to_string() to ensure all rows are printed.
    else:
        print("Could not generate highlight scenes.")

if __name__ == "__main__":
    main()