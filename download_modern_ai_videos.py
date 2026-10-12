"""
download_modern_ai_videos.py
Downloads ~300 modern AI-generated videos from luchaoqi/TalkingHeadBench on Hugging Face:
- LivePortrait (50)
- Hallo2 (50)
- Hallo (50)
- AniPortraitAudio (50)
- AniPortraitVideo (50)
- EmoPortrait (50)
Saves directly into data/combined/fake/
"""

import os
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from huggingface_hub import HfApi, hf_hub_download

TARGET_DIR = Path("data/combined/fake")
TARGET_DIR.mkdir(parents=True, exist_ok=True)

CATEGORIES = [
    "LivePortrait",
    "Hallo2",
    "Hallo",
    "AniPortraitAudio",
    "AniPortraitVideo",
    "EmoPortrait"
]
SAMPLES_PER_CATEGORY = 50

def main():
    print("Fetching repository index from luchaoqi/TalkingHeadBench...")
    api = HfApi()
    repo_files = list(api.list_repo_files(repo_id="luchaoqi/TalkingHeadBench", repo_type="dataset"))
    
    existing_files = set(f.name for f in TARGET_DIR.iterdir() if f.is_file())
    
    tasks_to_download = []
    
    for cat in CATEGORIES:
        matching = [
            f for f in repo_files
            if f.endswith(".mp4") and f.startswith(f"fake/{cat}/")
        ]
        
        count = 0
        for f in matching:
            dest_name = f"thb_{cat}_{Path(f).name}"
            if dest_name in existing_files:
                continue
            tasks_to_download.append((cat, f, dest_name))
            count += 1
            if count >= SAMPLES_PER_CATEGORY:
                break
        print(f"[{cat}] Queued {count} new videos to download (total pool: {len(matching)}).")
        
    print(f"\nTotal videos to download: {len(tasks_to_download)}")
    if not tasks_to_download:
        print("All target videos already exist!")
        return

    def download_and_save(item):
        cat, repo_path, dest_name = item
        dest_path = TARGET_DIR / dest_name
        cached_path = hf_hub_download(
            repo_id="luchaoqi/TalkingHeadBench",
            filename=repo_path,
            repo_type="dataset"
        )
        shutil.copy2(cached_path, dest_path)
        return dest_name

    completed = 0
    errors = 0
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(download_and_save, item): item for item in tasks_to_download}
        for future in as_completed(futures):
            item = futures[future]
            try:
                dest_name = future.result()
                completed += 1
                if completed % 25 == 0 or completed == len(tasks_to_download):
                    print(f"Progress: [{completed}/{len(tasks_to_download)}] downloaded...")
            except Exception as e:
                errors += 1
                print(f"Error downloading {item[1]}: {e}")

    print(f"\nFinished download! Success: {completed}, Errors: {errors}")
    print(f"Total videos currently in {TARGET_DIR}: {len(list(TARGET_DIR.glob('*.mp4')))}")

if __name__ == "__main__":
    main()
