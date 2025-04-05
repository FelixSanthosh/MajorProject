from coco_labels import COCO_LABELS

def map_ids_to_labels(detections):
    """Convert detected category IDs to names."""
    return [(COCO_LABELS.get(obj_id, "Unknown"), score) for obj_id, score in detections]

# Example detections (from your model)
detections = [(1, 0.98), (3, 0.85), (18, 0.76)]  # (category_id, confidence_score)

# Convert IDs to labels
labeled_detections = map_ids_to_labels(detections)
print(labeled_detections)  # [('person', 0.98), ('car', 0.85), ('dog', 0.76)]
