import os
import shutil
import json
import time

UPLOAD_DIR = r"E:\obj\uploaded_images"
CATEGORIZED_DIR = r"E:\obj\filtered_images"
ANNOTATIONS_PATH = r"E:\obj\dataset\annotations\instances_train2017.json"
BATCH_SIZE = 100  # Run models after 100 images

# ✅ Load COCO categories
with open(ANNOTATIONS_PATH, "r") as f:
    coco_data = json.load(f)
categories = {cat["id"]: cat["name"] for cat in coco_data["categories"]}

# ✅ Ensure category folders exist
for category in categories.values():
    os.makedirs(os.path.join(CATEGORIZED_DIR, category), exist_ok=True)

def categorize_images():
    uploaded_files = os.listdir(UPLOAD_DIR)
    
    if len(uploaded_files) >= BATCH_SIZE:
        print(f"✅ {len(uploaded_files)} images uploaded. Categorizing...")

        for file in uploaded_files:
            category = get_image_category(file)  # 🔹 Replace this with ML-based classifier
            src_path = os.path.join(UPLOAD_DIR, file)
            dest_path = os.path.join(CATEGORIZED_DIR, category, file)

            if os.path.exists(src_path):
                shutil.move(src_path, dest_path)

        print("✅ Images categorized successfully. Running models...")
        run_models()
    else:
        print(f"ℹ️ {len(uploaded_files)} images uploaded. Waiting for {BATCH_SIZE}.")

def get_image_category(image_name):
    """
    Dummy function: Replace this with an AI model or ML-based classifier.
    For now, it categorizes based on file naming convention.
    """
    if "car" in image_name.lower():
        return "car"
    elif "plane" in image_name.lower():
        return "airplane"
    elif "dog" in image_name.lower():
        return "dog"
    else:
        return "unknown"

def run_models():
    os.system("python E:\\obj\\app.py")  # Calls main app to run CNN & ViT
