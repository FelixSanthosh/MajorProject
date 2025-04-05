import os
import json

# ✅ Paths (Update if necessary)
ANNOTATIONS_PATH = r"E:\obj\dataset\annotations\instances_train2017.json"
IMAGES_DIR = r"E:\obj\dataset\train2017\train2017"
OUTPUT_DIR = r"E:\obj\filtered_images"

# ✅ Load COCO annotations
with open(ANNOTATIONS_PATH, "r") as f:
    coco_data = json.load(f)

# ✅ Load category mapping
categories = {cat["id"]: cat["name"] for cat in coco_data["categories"]}

# ✅ Ensure category folders exist
for category in categories.values():
    category_path = os.path.join(OUTPUT_DIR, category)
    os.makedirs(category_path, exist_ok=True)

# ✅ Map image IDs to filenames
image_id_to_filename = {img["id"]: img["file_name"] for img in coco_data["images"]}

# ✅ Process and move images
for annotation in coco_data["annotations"]:
    image_id = annotation["image_id"]
    category_id = annotation["category_id"]

    image_filename = image_id_to_filename.get(image_id, None)
    category_name = categories.get(category_id, None)

    if image_filename and category_name:
        src_path = os.path.join(IMAGES_DIR, image_filename)
        dest_path = os.path.join(OUTPUT_DIR, category_name, image_filename)

        if os.path.exists(src_path):  # ✅ Ensure image exists before moving
            try:
                os.rename(src_path, dest_path)  # ✅ Use rename() to cut instead of copy
            except Exception as e:
                print(f"Error moving {src_path} → {dest_path}: {e}")
        else:
            print(f"Warning: Image {src_path} not found.")

print("✅ Dataset categorized and moved successfully! 🚀")
