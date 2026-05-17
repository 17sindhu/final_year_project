import os
import gdown

FOLDER_ID = "16qDTPcu64V1b4_8n9qXfRV3r6iCzmkok"

if not os.path.exists("models"):
    os.makedirs("models", exist_ok=True)

    print("Downloading models folder...")

    gdown.download_folder(
        id=FOLDER_ID,
        output="models",
        quiet=False,
        use_cookies=False
    )

print("Models ready.")