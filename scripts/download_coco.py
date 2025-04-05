import os
import requests
from pycocotools.coco import COCO

# Define dataset directory
coco_dir = "E:/obj/dataset"
ann_file = os.path.join(coco_dir, "annotations/instances_train2017.json")

# Load COCO dataset
coco = COCO(ann_file)

# Define categories to download
selected_categories = ["person", "car", "bicycle"]
category_ids = coco.getCatIds(catNms=selected_categories)

# Get image IDs
image_ids = set()
for cat_id in category_ids:
    image_ids.update(coco.getImgIds(catIds=cat_id))
image_ids = list(image_ids)

print(f"Found {len(image_ids)} images for categories {selected_categories}")

# Directory to save images
filtered_img_dir = os.path.join(coco_dir, "../filtered_images")
os.makedirs(filtered_img_dir, exist_ok=True)

# Function to download images
def download_images(img_id):
    img_info = coco.loadImgs(img_id)[0]
    img_url = img_info["coco_url"]
    
    img_data = requests.get(img_url).content
    img_path = os.path.join(filtered_img_dir, img_info["file_name"])
    
    with open(img_path, "wb") as f:
        f.write(img_data)
    print(f"Downloaded: {img_path}")

# Download first 10 images
for img_id in image_ids[:10]:  
    download_images(img_id)

print("✅ Image Download Completed!")
