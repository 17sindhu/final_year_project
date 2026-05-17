"""
download_models.py
Downloads model files from Google Drive on first startup.
Replace FILE_ID values with your actual Google Drive file IDs.
"""

import os
import gdown

# ── Replace these with your actual Google Drive file IDs ─────────────────────
# Get file ID from share link:
# https://drive.google.com/file/d/FILE_ID_HERE/view
#                                  ^^^^^^^^^^^^

MODEL_FILES = {
    "models/svm_model.pkl":                    "PASTE_FILE_ID_1_HERE",
    "models/tfidf_vectorizer.pkl":             "PASTE_FILE_ID_2_HERE",
    "models/bilstm_model.h5":                  "PASTE_FILE_ID_3_HERE",
    "models/bilstm_tokenizer.pkl":             "PASTE_FILE_ID_4_HERE",
    "models/bilstm_label_encoder (1).pkl":     "PASTE_FILE_ID_5_HERE",
    "models/mbert_model.pkl":                  "PASTE_FILE_ID_6_HERE",
    "dictionary.csv":                          "PASTE_FILE_ID_7_HERE",
}

def download_all():
    """Download all model files from Google Drive if not already present."""

    os.makedirs("models", exist_ok=True)

    all_ok = True
    for local_path, file_id in MODEL_FILES.items():

        # Skip if file already downloaded
        if os.path.exists(local_path):
            print(f"[SKIP] {local_path} already exists.")
            continue

        # Skip placeholder IDs
        if "PASTE_FILE_ID" in file_id:
            print(f"[WARN] No file ID set for {local_path} — skipping.")
            continue

        print(f"[DOWN] Downloading {local_path} ...")
        try:
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, local_path, quiet=False, fuzzy=True)

            if os.path.exists(local_path):
                size = os.path.getsize(local_path) / (1024 * 1024)
                print(f"[OK]   {local_path} downloaded ({size:.1f} MB)")
            else:
                print(f"[ERR]  {local_path} download failed.")
                all_ok = False

        except Exception as e:
            print(f"[ERR]  Failed to download {local_path}: {e}")
            all_ok = False

    if all_ok:
        print("[INFO] All model files ready.")
    else:
        print("[WARN] Some files failed — app will run in demo mode.")


if __name__ == "__main__":
    download_all()
