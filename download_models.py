import os
import gdown

MODELS = {

    "models/svm_model.pkl":
    "17j-kqNT-Jw5bsNS28sDYHJUR0iHWSqzF",

    "models/tfidf_vectorizer.pkl":
    "1b4qtZm1u0EQb2bwCn3QHu1SjnW4x4m9b"
}

os.makedirs("models", exist_ok=True)

for path, file_id in MODELS.items():

    if not os.path.exists(path):

        url = f"https://drive.google.com/uc?id={file_id}"

        print(f"Downloading {path}...")

        gdown.download(url, path, quiet=False)

print("Models downloaded successfully")