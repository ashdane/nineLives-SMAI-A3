import os
import sys
import shutil
import zipfile
import urllib.request

# Global paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "tomato")
TEMP_DIR = os.path.join(BASE_DIR, "data", "_temp")

HF_ZIP_URL = "https://huggingface.co/datasets/mohanty/PlantVillage/resolve/main/data.zip"

TOMATO_CLASSES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

def download_progress(count, block_size, total_size):
    # Just print a simple progress
    downloaded = count * block_size
    if total_size > 0:
        percent = int(downloaded * 100 / total_size)
        percent = min(100, percent)
        sys.stdout.write(f"\rDownloading... {percent}%")
        sys.stdout.flush()
    else:
        sys.stdout.write(f"\rDownloaded {downloaded} bytes")
        sys.stdout.flush()

def main():
    if os.path.exists(DATA_DIR) and len(os.listdir(DATA_DIR)) > 0:
        print("Data is already downloaded in", DATA_DIR)
        print("Delete the folder first if you want to redownload.")
        return

    os.makedirs(TEMP_DIR, exist_ok=True)
    zip_path = os.path.join(TEMP_DIR, "data.zip")

    if not os.path.exists(zip_path):
        print("Starting download from huggingface:", HF_ZIP_URL)
        urllib.request.urlretrieve(HF_ZIP_URL, zip_path, reporthook=download_progress)
        print("\nDownload finished.")
    else:
        print("Found zip file already at", zip_path)

    # extract step
    print("Extracting zip file...")
    extract_dir = os.path.join(TEMP_DIR, "extracted")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(extract_dir)
    print("Done extracting.")

    # find where the color folder is
    color_dir = None
    for root, dirs, files in os.walk(extract_dir):
        if "color" in dirs:
            cand = os.path.join(root, "color")
            # checking if it has tomato
            has_tomato = False
            for d in os.listdir(cand):
                if d.lower().startswith("tomato"):
                    has_tomato = True
                    break
            if has_tomato:
                color_dir = cand
                break
        
        # fallback
        tomato_count = 0
        for d in dirs:
            if d.lower().startswith("tomato"):
                tomato_count += 1
        if tomato_count >= 5:
            color_dir = root
            break

    if color_dir is None:
        for root, dirs, files in os.walk(extract_dir):
            for d in dirs:
                if d in TOMATO_CLASSES:
                    color_dir = root
                    break
            if color_dir:
                break

    if not color_dir:
        print("Error: Cant find tomato folders inside the extracted zip.")
        sys.exit(1)

    print("Found the plant folders here:", color_dir)
    os.makedirs(DATA_DIR, exist_ok=True)

    saved = 0
    counts = {}

    for c in TOMATO_CLASSES:
        src = os.path.join(color_dir, c)
        if not os.path.exists(src):
            print("Warning: Missing class folder", c)
            continue

        dst = os.path.join(DATA_DIR, c)
        # copy tree
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        
        # count files
        num_files = len(os.listdir(dst))
        counts[c] = num_files
        saved += num_files

    print("\nSaved total tomato images:", saved)
    print("Breakdown per class:")
    for k, v in counts.items():
        print(f" - {k}: {v}")

    print("Cleaning up temp stuff...")
    try:
        shutil.rmtree(extract_dir)
    except:
        pass
    print("Done downloading data.")

if __name__ == "__main__":
    main()
