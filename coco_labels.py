# import json

# def load_coco_labels(json_path="data/instances_val2017.json"):
#     """Load COCO category labels from JSON file."""
#     with open(json_path, "r") as f:
#         coco_data = json.load(f)
    
#     categories = coco_data["categories"]
#     return {cat["id"]: cat["name"] for cat in categories}

# # Load labels once
# COCO_LABELS = load_coco_labels()

import json

def load_coco_labels(json_path):
    """Load COCO category mappings from a JSON file."""
    with open(json_path, "r") as f:
        coco_data = json.load(f)
    
    # Extract category ID-to-name mapping
    category_mapping = {cat["id"]: cat["name"] for cat in coco_data["categories"]}
    return category_mapping

# Example usage
if __name__ == "__main__":
    coco_labels = load_coco_labels("../dataset/annotations/instances_val2017.json")
    print(coco_labels)  # Check output
