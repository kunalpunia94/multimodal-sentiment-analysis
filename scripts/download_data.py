"""Dataset acquisition helper.

This script prints step-by-step instructions for downloading and organizing
CREMA-D audio/video files using Kaggle CLI.
"""

import os


def main():
    """Print CLI commands to download and validate dataset folders.

    Returns:
        None
    """
    print("--- CREMA-D Dataset Download Guide (Kaggle) ---")
    print("\nThis script does not download data automatically.")
    print("Follow these steps:\n")

    print("1. Install Kaggle CLI in your active environment:")
    print("   uv pip install kaggle")

    print("\n2. Create Kaggle API token from Account settings and export it:")
    print("   export KAGGLE_API_TOKEN='<your_token_here>'")

    print("\n3. Download audio and extract:")
    print("   kaggle datasets download -d ejlok1/cremad")
    print("   unzip -o cremad.zip -d data")

    print("\n4. Download videos and collect into VideoFlash:")
    print("   kaggle datasets download -d yassmenyoussef/cremad-mp4")
    print("   unzip -o cremad-mp4.zip -d data")
    print("   mkdir -p data/VideoFlash")
    print("   find data -type f -name '*.mp4' -exec cp {} data/VideoFlash/ \\\;")

    print("\n5. Optional cleanup:")
    print("   rm -rf data/output_videos")
    print("   rm -f cremad.zip cremad-mp4.zip")

    print("\nExpected structure:")
    print("data/")
    print("├── AudioWAV/")
    print("└── VideoFlash/")

    audio_path = os.path.join("data", "AudioWAV")
    video_path = os.path.join("data", "VideoFlash")

    if os.path.exists(audio_path) and os.path.exists(video_path):
        print("\nValidation: Found data/AudioWAV and data/VideoFlash directories.")
    else:
        print("\nValidation: data/AudioWAV and/or data/VideoFlash not found yet.")


if __name__ == "__main__":
    main()
