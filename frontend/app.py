# #claudency26/03
import streamlit as st
from PIL import Image, UnidentifiedImageError
import sys
import os
import time
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Set Streamlit page configuration
st.set_page_config(page_title="Object Detection: CNN vs ViT", layout="wide")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from coco_labels import load_coco_labels
from models.cnn_model import detect_objects_cnn, get_cnn_training_loss
from models.vit_model import detect_objects_vit, get_vit_training_loss

# Title
st.title("🔍 Object Detection: CNN vs ViT")

# Sidebar options
st.sidebar.header("Settings")
run_cnn = st.sidebar.checkbox("Run CNN Model", value=True)
run_vit = st.sidebar.checkbox("Run ViT Model", value=True)

# Multi-file uploader
uploaded_files = st.file_uploader("📂 Upload up to 100 images", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    num_files = len(uploaded_files)
    st.write(f"✅ Uploaded {num_files} image(s).")

    if num_files > 100:
        st.warning("⚠ You uploaded more than 100 images. Only the first 100 will be processed.")
        uploaded_files = uploaded_files[:100]

    # Load COCO labels
    coco_labels = load_coco_labels("dataset/annotations/instances_train2017.json")
    coco_labels.update(load_coco_labels("dataset/annotations/instances_val2017.json"))

    # Helper function to format detection results
    def format_results(results):
        formatted_results = []
        for item in results:
            obj_id = item.get("Object") or item.get("Label")  
            obj_label = coco_labels.get(obj_id, f"Unknown ({obj_id})")
            confidence = item.get("Image confidence", 0)
            formatted_results.append((obj_label, confidence))  
        return formatted_results

    # Categorization storage
    category_images = defaultdict(list)
    category_results = defaultdict(lambda: {"cnn": [], "vit": []})
    category_image_details = defaultdict(list)
    
    # Performance tracking
    cnn_times, vit_times = [], []
    cnn_flops_list, vit_flops_list = [], []
    cnn_losses, vit_losses = get_cnn_training_loss(), get_vit_training_loss()
    
    # For tracking overall metrics
    cnn_confidences, vit_confidences = [], []
    cnn_accuracies, vit_accuracies = [], []  # Lists for tracking model accuracies
    
    cnn_count, vit_count = 0, 0
    cnn_total_objects, vit_total_objects = 0, 0  
    category_counts = defaultdict(lambda: {"cnn": 0, "vit": 0})

    # Progress bar
    progress_bar = st.progress(0)

    # Process each image
    for idx, uploaded_file in enumerate(uploaded_files):
        st.write(f"### 🖼 Image {idx+1}: {uploaded_file.name}")

        try:
            image = Image.open(uploaded_file).convert("RGB")  
        except UnidentifiedImageError:
            st.error(f"❌ Error: Unable to open {uploaded_file.name}. Skipping...")
            continue

        st.write("🚀 Running Object Detection...")

        detected_objects = set()
        cnn_image, vit_image = None, None
        
        # Store original results before formatting
        raw_cnn_results, raw_vit_results = [], []
        
        # CNN Model Processing
        if run_cnn:
            raw_cnn_results, cnn_time, cnn_image, cnn_avg_conf, cnn_accuracy, cnn_flops = detect_objects_cnn(image.copy())
            # Ensure accuracy is never greater than 1.0 (100%)
            cnn_accuracy = min(cnn_accuracy, 1.0)
            cnn_times.append(cnn_time)
            cnn_confidences.append(cnn_avg_conf)
            cnn_accuracies.append(cnn_accuracy)  # Store the accuracy from the model
            cnn_flops_list.append(cnn_flops)
            cnn_count += 1
            cnn_results = format_results(raw_cnn_results)
            cnn_total_objects += len(cnn_results)

        # ViT Model Processing
        if run_vit:
            raw_vit_results, vit_time, vit_image, vit_avg_conf, vit_accuracy, vit_flops = detect_objects_vit(image.copy())
            # Ensure accuracy is never greater than 1.0 (100%)
            vit_accuracy = min(vit_accuracy, 1.0)
            vit_times.append(vit_time)
            vit_confidences.append(vit_avg_conf)
            vit_accuracies.append(vit_accuracy)  # Store the accuracy from the model
            vit_flops_list.append(vit_flops)
            vit_count += 1
            vit_results = format_results(raw_vit_results)
            vit_total_objects += len(vit_results)

        # Display images in the same row
        col1, col2, col3 = st.columns(3)
        col1.image(image, caption="Original Image", use_container_width=True)
        if run_cnn and cnn_image is not None:
            col2.image(cnn_image, caption="CNN Detection", use_container_width=True)
        if run_vit and vit_image is not None:
            col3.image(vit_image, caption="ViT Detection", use_container_width=True)

        # Create dictionaries to store object-specific metrics for this image
        image_object_confidences = {}
        image_object_accuracies = {}
        
        # Process CNN detections
        if run_cnn:
            for obj, confidence in cnn_results:
                detected_objects.add(obj)
                # Store confidence and time for each object
                category_results[obj]["cnn"].append((confidence, cnn_time))
                category_counts[obj]["cnn"] += 1
                
                # Store object-specific metrics
                if obj not in image_object_confidences:
                    image_object_confidences[obj] = {"cnn_confidence": confidence, "vit_confidence": None}
                else:
                    image_object_confidences[obj]["cnn_confidence"] = confidence
                    
                # FIXED: Remove randomization - use confidence to adjust accuracy instead
                # This ensures consistency and prevents biased results
                confidence_factor = max(0.8, min(confidence, 1.0))  # Scale confidence to reasonable range
                object_accuracy = min(cnn_accuracy * confidence_factor, 1.0)
                
                if obj not in image_object_accuracies:
                    image_object_accuracies[obj] = {"cnn_accuracy": object_accuracy, "vit_accuracy": None}
                else:
                    image_object_accuracies[obj]["cnn_accuracy"] = object_accuracy
                    
        # Process ViT detections
        if run_vit:
            for obj, confidence in vit_results:
                detected_objects.add(obj)
                # Store confidence and time for each object
                category_results[obj]["vit"].append((confidence, vit_time))
                category_counts[obj]["vit"] += 1
                
                # Store object-specific confidence
                if obj not in image_object_confidences:
                    image_object_confidences[obj] = {"cnn_confidence": None, "vit_confidence": confidence}
                else:
                    image_object_confidences[obj]["vit_confidence"] = confidence
                    
                # FIXED: Remove randomization - use confidence to adjust accuracy instead
                # Apply same logic to ViT for fairness
                confidence_factor = max(0.8, min(confidence, 1.0))  # Scale confidence to reasonable range
                object_accuracy = min(vit_accuracy * confidence_factor, 1.0)
                
                if obj not in image_object_accuracies:
                    image_object_accuracies[obj] = {"cnn_accuracy": None, "vit_accuracy": object_accuracy}
                else:
                    image_object_accuracies[obj]["vit_accuracy"] = object_accuracy
                    
        # Add image and object details to each detected category
        for obj in detected_objects:
            # Get object-specific metrics
            obj_confidences = image_object_confidences.get(obj, {"cnn_confidence": None, "vit_confidence": None})
            obj_accuracies = image_object_accuracies.get(obj, {"cnn_accuracy": None, "vit_accuracy": None})
            
            # Store the image with its detection data
            category_images[obj].append((uploaded_file, cnn_image, vit_image))
            
            # Store object-specific metrics for this image - use different values for accuracy and confidence
            category_image_details[obj].append({
                "image": uploaded_file.name,
                "cnn_accuracy": obj_accuracies["cnn_accuracy"] if run_cnn and obj_accuracies["cnn_accuracy"] is not None else None,
                "cnn_confidence": obj_confidences["cnn_confidence"] if run_cnn and obj_confidences["cnn_confidence"] is not None else None,
                "vit_accuracy": obj_accuracies["vit_accuracy"] if run_vit and obj_accuracies["vit_accuracy"] is not None else None,
                "vit_confidence": obj_confidences["vit_confidence"] if run_vit and obj_confidences["vit_confidence"] is not None else None
            })

        progress_bar.progress((idx + 1) / num_files)

    # Sidebar to select categories
    st.sidebar.header("Select Category")
    category_selected = st.sidebar.selectbox("Choose a category:", list(category_images.keys()))

    if category_selected:
        cnn_objects = category_counts[category_selected]["cnn"]
        vit_objects = category_counts[category_selected]["vit"]

        st.subheader(f"📊 *Objects detected in '{category_selected}' Category:*")
        for img, cnn_img, vit_img in category_images[category_selected]:
            col1, col2, col3 = st.columns(3)
            col1.image(img, caption="Original Image", use_container_width=True)
            if run_cnn and cnn_img is not None:
                col2.image(cnn_img, caption="CNN Detection", use_container_width=True)
            if run_vit and vit_img is not None:
                col3.image(vit_img, caption="ViT Detection", use_container_width=True)

    # Performance Comparison Graphs
    st.write("## 📊 Model Comparison Graphs")
    
    # Training Loss Graph
    fig, ax = plt.subplots()
    ax.plot(range(1, len(cnn_losses) + 1), cnn_losses, label="CNN (ResNet-50)", marker="o", color="blue")
    ax.plot(range(1, len(vit_losses) + 1), vit_losses, label="ViT (ViT-Base)", marker="s", color="orange")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Training Loss")
    ax.set_title("Training Loss Over Time")
    ax.legend()
    st.pyplot(fig)

    # Inference Time Graph
    fig, ax = plt.subplots()
    ax.bar(["CNN", "ViT"], [np.mean(cnn_times) if cnn_times else 0, np.mean(vit_times) if vit_times else 0], color=["blue", "orange"])
    ax.set_ylabel("Avg Inference Time (ms)")
    ax.set_title("Inference Time per Image")
    st.pyplot(fig)

    # FLOPs Graph
    fig, ax = plt.subplots()
    ax.bar(["CNN (ResNet-50)", "ViT (ViT-Base)"], 
           [np.mean(cnn_flops_list) if cnn_flops_list else 0, np.mean(vit_flops_list) if vit_flops_list else 0], 
           color=["blue", "orange"])
    ax.set_ylabel("FLOPs (Floating Point Operations)")
    ax.set_title("FLOPs Comparison")
    st.pyplot(fig)
   
    # Compute overall metrics using the correct values
    avg_cnn_conf = np.mean(cnn_confidences) if cnn_confidences else 0
    avg_vit_conf = np.mean(vit_confidences) if vit_confidences else 0
    
    # Use the separate accuracy values from the models
    # Ensure accuracy is never greater than 100%
    avg_cnn_acc = min(np.mean(cnn_accuracies) * 100, 100.0) if cnn_accuracies else 0
    avg_vit_acc = min(np.mean(vit_accuracies) * 100, 100.0) if vit_accuracies else 0
    
    # Calculate category-specific metrics
    cnn_category_confidences = []
    vit_category_confidences = []
    cnn_category_accuracies = []
    vit_category_accuracies = []
    
    # Check for data availability before calculating category metrics
    category_comparison_valid = True
    
    if category_selected:
        cnn_category_confidences = [conf for conf, _ in category_results[category_selected]["cnn"]]
        vit_category_confidences = [conf for conf, _ in category_results[category_selected]["vit"]]
        
        # Extract per-image accuracies for the selected category
        for details in category_image_details[category_selected]:
            if details["cnn_accuracy"] is not None:
                cnn_category_accuracies.append(details["cnn_accuracy"])
            if details["vit_accuracy"] is not None:
                vit_category_accuracies.append(details["vit_accuracy"])

        # Check if both models have data for the selected category
        if len(cnn_category_accuracies) == 0 or len(vit_category_accuracies) == 0:
            category_comparison_valid = False
            st.warning(f"⚠️ Not enough data for both models in '{category_selected}' category for a fair comparison.")

        avg_cnn_category_conf = np.mean(cnn_category_confidences) if cnn_category_confidences else 0
        avg_vit_category_conf = np.mean(vit_category_confidences) if vit_category_confidences else 0
        
        # Calculate average category-specific accuracy
        avg_cnn_category_acc = np.mean(cnn_category_accuracies) * 100 if cnn_category_accuracies else 0
        avg_vit_category_acc = np.mean(vit_category_accuracies) * 100 if vit_category_accuracies else 0

    else:
        avg_cnn_category_conf = 0
        avg_vit_category_conf = 0
        avg_cnn_category_acc = 0
        avg_vit_category_acc = 0

    # Calculate inference time metrics
    avg_cnn_time = np.mean(cnn_times) if cnn_times else 0
    avg_vit_time = np.mean(vit_times) if vit_times else 0
    
    # Calculate FLOPs metrics
    avg_cnn_flops = np.mean(cnn_flops_list) if cnn_flops_list else 0
    avg_vit_flops = np.mean(vit_flops_list) if vit_flops_list else 0
    
    # Calculate training loss final values
    cnn_final_loss = cnn_losses[-1] if cnn_losses else 0
    vit_final_loss = vit_losses[-1] if vit_losses else 0

    # Initialize counters for model comparison
    cnn_wins_overall = 0
    vit_wins_overall = 0
    tie_count_overall = 0
    
    cnn_wins_category = 0
    vit_wins_category = 0
    tie_count_category = 0
    
    # Compare overall metrics (only the 3 specified metrics)
    # 1. Total Objects Detected
    if cnn_total_objects > vit_total_objects:
        cnn_wins_overall += 1
    elif vit_total_objects > cnn_total_objects:
        vit_wins_overall += 1
    else:
        tie_count_overall += 1
    
    # 2. Overall Accuracy
    if avg_cnn_acc > avg_vit_acc:
        cnn_wins_overall += 1
    elif avg_vit_acc > avg_cnn_acc:
        vit_wins_overall += 1
    else:
        tie_count_overall += 1
    
    # 3. Overall Confidence
    if avg_cnn_conf > avg_vit_conf:
        cnn_wins_overall += 1
    elif avg_vit_conf > avg_cnn_conf:
        vit_wins_overall += 1
    else:
        tie_count_overall += 1
    
    # Determine the best model overall
    if cnn_wins_overall > vit_wins_overall:
        best_model_overall = "CNN"
    elif vit_wins_overall > cnn_wins_overall:
        best_model_overall = "ViT"
    else:
        # If there's a tie, use accuracy as tiebreaker
        if avg_cnn_acc > avg_vit_acc:
            best_model_overall = "CNN"
        elif avg_vit_acc > avg_cnn_acc:
            best_model_overall = "ViT"
        else:
            # If still tied, use confidence
            if avg_cnn_conf > avg_vit_conf:
                best_model_overall = "CNN"
            else:
                best_model_overall = "ViT"
    
    # Compare category-specific metrics if a category is selected and valid
    if category_selected and category_comparison_valid:
        # 1. Objects detected in category
        if cnn_objects > vit_objects:
            cnn_wins_category += 1
        elif vit_objects > cnn_objects:
            vit_wins_category += 1
        else:
            tie_count_category += 1
        
        # 2. Accuracy in category
        if avg_cnn_category_acc > avg_vit_category_acc:
            cnn_wins_category += 1
        elif avg_vit_category_acc > avg_cnn_category_acc:
            vit_wins_category += 1
        else:
            tie_count_category += 1
        
        # 3. Confidence in category
        if avg_cnn_category_conf > avg_vit_category_conf:
            cnn_wins_category += 1
        elif avg_vit_category_conf > avg_cnn_category_conf:
            vit_wins_category += 1
        else:
            tie_count_category += 1
        
        # 4. Inference Time in category
        if avg_cnn_time < avg_vit_time:
            cnn_wins_category += 1
        elif avg_vit_time < avg_cnn_time:
            vit_wins_category += 1
        else:
            tie_count_category += 1
        
        # 5. FLOPs in category
        if avg_cnn_flops < avg_vit_flops:
            cnn_wins_category += 1
        elif avg_vit_flops < avg_cnn_flops:
            vit_wins_category += 1
        else:
            tie_count_category += 1
        
        # 6. Training Loss in category
        if cnn_final_loss < vit_final_loss:
            cnn_wins_category += 1
        elif vit_final_loss < cnn_final_loss:
            vit_wins_category += 1
        else:
            tie_count_category += 1
        
        # Determine the best model for category
        if cnn_wins_category > vit_wins_category:
            best_model_category = "CNN"
        elif vit_wins_category > cnn_wins_category:
            best_model_category = "ViT"
        else:
            # If tied, use accuracy as tiebreaker
            if avg_cnn_category_acc > avg_vit_category_acc:
                best_model_category = "CNN"
            elif avg_vit_category_acc > avg_cnn_category_acc:
                best_model_category = "ViT"
            else:
                # If still tied, use confidence
                if avg_cnn_category_conf > avg_vit_category_conf:
                    best_model_category = "CNN"
                else:
                    best_model_category = "ViT"
    else:
        best_model_category = "N/A"
    
    # Calculate confidence levels
    confidence_margin_overall = abs(cnn_wins_overall - vit_wins_overall) / 3  # 3 metrics compared
    if confidence_margin_overall > 0.5:
        confidence_level_overall = "high confidence"
    elif confidence_margin_overall > 0.2:
        confidence_level_overall = "moderate confidence"
    else:
        confidence_level_overall = "slight edge"
    
    # Add confidence level for category-specific conclusion
    confidence_level_category = "N/A"
    if category_selected and category_comparison_valid:
        confidence_margin_category = abs(cnn_wins_category - vit_wins_category) / 6  # 6 metrics compared
        if confidence_margin_category > 0.5:
            confidence_level_category = "high confidence"
        elif confidence_margin_category > 0.2:
            confidence_level_category = "moderate confidence"
        else:
            confidence_level_category = "slight edge"
    
    st.subheader("📊 *RESULT METRICS*")
    
    # Prepare the results data for the table
    results_data = {
        "Metric": [
            "Total Objects Detected",
            "Overall Accuracy (%)",
            "Overall Confidence",
            f"Objects detected in '{category_selected}' Category" if category_selected else "Category Objects",
            f"Accuracy(%) in '{category_selected}' Category" if category_selected else "Category Accuracy",
            f"Confidence in '{category_selected}' Category" if category_selected else "Category Confidence",
            f"Inference Time (ms) in '{category_selected}' Category" if category_selected else "Category Inference Time",
            f"FLOPs in '{category_selected}' Category" if category_selected else "Category FLOPs",
            f"Training Loss in '{category_selected}' Category" if category_selected else "Category Training Loss",
            "🏆 Best Model Overall",
            f"🏆 Best Model for '{category_selected}' Category" if category_selected else "🏆 Best Model for Category"
        ],
        "CNN": [
            cnn_total_objects,
            f"{avg_cnn_acc:.2f}%",
            f"{avg_cnn_conf:.2f}",
            f"{cnn_objects} objects" if category_selected else "-",
            f"{avg_cnn_category_acc:.2f}%" if category_selected else "-",
            f"{avg_cnn_category_conf:.4f}" if category_selected else "-",
            f"{avg_cnn_time:.2f}" if category_selected else "-",
            f"{avg_cnn_flops:.2f}" if category_selected else "-",
            f"{cnn_final_loss:.2f}" if category_selected else "-",
            "✓" if best_model_overall == "CNN" else "-",
            "✓" if category_selected and best_model_category == "CNN" else "-"
        ],
         "ViT": [
            vit_total_objects,
            f"{avg_vit_acc:.2f}%",
            f"{avg_vit_conf:.2f}",
            f"{vit_objects} objects" if category_selected else "-",
            f"{avg_vit_category_acc:.2f}%" if category_selected else "-",
            f"{avg_vit_category_conf:.4f}" if category_selected else "-",
            f"{avg_vit_time:.2f}" if category_selected else "-",
            f"{avg_vit_flops:.2f}" if category_selected else "-",
            f"{vit_final_loss:.2f}" if category_selected else "-",
            "✓" if best_model_overall == "ViT" else "-",
            "✓" if category_selected and best_model_category == "ViT" else "-"
        ]
    }
    
    # Display the table
    st.table(pd.DataFrame(results_data))

    # Display model comparison details with the same metrics as RESULT METRICS
    with st.expander("📊 *Detailed Model Comparison*"):
        st.write("### Model Comparison Details")
        st.write("The best model is determined by comparing key performance indicators:")
        
        # Create a simple explanation of how the best model was chosen
        comparison_text = f"""
        ## Overall Comparison
        
        ### Overall Metrics Summary
        - For overall comparison, we focused on these 3 metrics:
          1. Total Objects Detected
          2. Overall Accuracy (%)
          3. Overall Confidence
          
        - CNN wins in {cnn_wins_overall} metrics
        - ViT wins in {vit_wins_overall} metrics
        
        ### Overall Metric-by-Metric Comparison:
        1. **Total Objects Detected**: {'CNN' if cnn_total_objects > vit_total_objects else 'ViT' if vit_total_objects > cnn_total_objects else 'Tie'} ({cnn_total_objects} vs {vit_total_objects})
        2. **Overall Accuracy**: {'CNN' if avg_cnn_acc > avg_vit_acc else 'ViT' if avg_vit_acc > avg_cnn_acc else 'Tie'} ({avg_cnn_acc:.2f}% vs {avg_vit_acc:.2f}%)
        3. **Overall Confidence**: {'CNN' if avg_cnn_conf > avg_vit_conf else 'ViT' if avg_vit_conf > avg_cnn_conf else 'Tie'} ({avg_cnn_conf:.2f} vs {avg_vit_conf:.2f})
        
        ### Best Model Overall: {best_model_overall} (with {confidence_level_overall})
        """
        
        # Add category-specific comparison section if a category is selected
        if category_selected and category_comparison_valid:
            category_comparison = f"""
            ## Category-Specific Comparison for '{category_selected}'
            
            ### Category Metrics Summary
            - For category comparison, we focused on these 6 metrics:
              1. Objects detected
              2. Accuracy (%)
              3. Confidence
              4. Inference Time (ms)
              5. FLOPs
              6. Training Loss
              
            - CNN wins in {cnn_wins_category} metrics
            - ViT wins in {vit_wins_category} metrics
            
            ### Category Metric-by-Metric Comparison:
            1. **Objects Detected**: {'CNN' if cnn_objects > vit_objects else 'ViT' if vit_objects > cnn_objects else 'Tie'} ({cnn_objects} vs {vit_objects})
            2. **Accuracy**: {'CNN' if avg_cnn_category_acc > avg_vit_category_acc else 'ViT' if avg_vit_category_acc > avg_cnn_category_acc else 'Tie'} ({avg_cnn_category_acc:.2f}% vs {avg_vit_category_acc:.2f}%)
            3. **Confidence**: {'CNN' if avg_cnn_category_conf > avg_vit_category_conf else 'ViT' if avg_vit_category_conf > avg_cnn_category_conf else 'Tie'} ({avg_cnn_category_conf:.4f} vs {avg_vit_category_conf:.4f})
            4. **Inference Time**: {'CNN' if avg_cnn_time < avg_vit_time else 'ViT' if avg_vit_time < avg_cnn_time else 'Tie'} ({avg_cnn_time:.2f}ms vs {avg_vit_time:.2f}ms) - lower is better
            5. **Computational Efficiency (FLOPs)**: {'CNN' if avg_cnn_flops < avg_vit_flops else 'ViT' if avg_vit_flops < avg_cnn_flops else 'Tie'} ({avg_cnn_flops:.2f} vs {avg_vit_flops:.2f}) - lower is better
            6. **Training Loss**: {'CNN' if cnn_final_loss < vit_final_loss else 'ViT' if vit_final_loss < cnn_final_loss else 'Tie'} ({cnn_final_loss:.2f} vs {vit_final_loss:.2f}) - lower is better
            
            ### Best Model for {category_selected}: {best_model_category} (with {confidence_level_category})
            """
            comparison_text += category_comparison
        else:
            comparison_text += "\n\n## Category-Specific Comparison\n\nNo category selected or not enough data for comparison."
            
        st.write(comparison_text)
        
        st.write(f"### Final Result")
        st.write(f"Based on winning in {cnn_wins_overall if best_model_overall == 'CNN' else vit_wins_overall} out of 3 key performance metrics, *{best_model_overall}* is the better model overall with {confidence_level_overall}.")
        
        if category_selected and category_comparison_valid:
            st.write(f"For the category '{category_selected}', *{best_model_category}* is the better model with {confidence_level_category}, based on winning in {cnn_wins_category if best_model_category == 'CNN' else vit_wins_category} out of 6 key category-specific metrics.")

    # Display Accuracy and Confidence
    if category_selected:
        st.subheader(f"📊 *Accuracy & Confidence per Image in '{category_selected}' Category*")
    
        # Only display details for images where this category was actually detected
        filtered_data = [
            {
                "Image": details["image"],
                "CNN Accuracy (%)": f"{details['cnn_accuracy'] * 100:.4f}%" if details["cnn_accuracy"] is not None else "-",
                "CNN Confidence": f"{details['cnn_confidence']:.4f}" if details["cnn_confidence"] is not None else "-",
                "ViT Accuracy (%)": f"{details['vit_accuracy'] * 100:.4f}%" if details["vit_accuracy"] is not None else "-",
                "ViT Confidence": f"{details['vit_confidence']:.4f}" if details["vit_confidence"] is not None else "-"
            }
            for details in category_image_details[category_selected]
        ]

        # Convert to Pandas DataFrame and display
        if filtered_data:
            st.table(pd.DataFrame(filtered_data))
        else:
            st.write("⚠ No data available for this category.")
# import streamlit as st
# from PIL import Image, UnidentifiedImageError
# import sys
# import os
# import time
# from collections import defaultdict
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd


# # Set Streamlit page configuration
# st.set_page_config(page_title="Object Detection: CNN vs ViT", layout="wide")

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# from coco_labels import load_coco_labels
# from models.cnn_model import detect_objects_cnn, get_cnn_training_loss
# from models.vit_model import detect_objects_vit, get_vit_training_loss

# # Title
# st.title("🔍 Object Detection: CNN vs ViT")

# # Sidebar options
# st.sidebar.header("Settings")
# run_cnn = st.sidebar.checkbox("Run CNN Model", value=True)
# run_vit = st.sidebar.checkbox("Run ViT Model", value=True)

# # Multi-file uploader
# uploaded_files = st.file_uploader("📂 Upload up to 100 images", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

# if uploaded_files:
#     num_files = len(uploaded_files)
#     st.write(f"✅ Uploaded {num_files} image(s).")

#     if num_files > 100:
#         st.warning("⚠ You uploaded more than 100 images. Only the first 100 will be processed.")
#         uploaded_files = uploaded_files[:100]

#     # Load COCO labels
#     coco_labels = load_coco_labels("dataset/annotations/instances_train2017.json")
#     coco_labels.update(load_coco_labels("dataset/annotations/instances_val2017.json"))

#     # Helper function to format detection results
#     def format_results(results):
#         formatted_results = []
#         for item in results:
#             obj_id = item.get("Object") or item.get("Label")  
#             obj_label = coco_labels.get(obj_id, f"Unknown ({obj_id})")
#             confidence = item.get("Image confidence", 0)
#             formatted_results.append((obj_label, confidence))  
#         return formatted_results

#     # Categorization storage
#     category_images = defaultdict(list)
#     category_results = defaultdict(lambda: {"cnn": [], "vit": []})
#     category_image_details = defaultdict(list)
    
#     # Performance tracking
#     cnn_times, vit_times = [], []
#     cnn_flops_list, vit_flops_list = [], []
#     cnn_losses, vit_losses = get_cnn_training_loss(), get_vit_training_loss()
    
#     # For tracking overall metrics
#     cnn_confidences, vit_confidences = [], []
#     cnn_accuracies, vit_accuracies = [], []  # Lists for tracking model accuracies
    
#     cnn_count, vit_count = 0, 0
#     cnn_total_objects, vit_total_objects = 0, 0  
#     category_counts = defaultdict(lambda: {"cnn": 0, "vit": 0})

#     # Progress bar
#     progress_bar = st.progress(0)

#     # Process each image
#     for idx, uploaded_file in enumerate(uploaded_files):
#         st.write(f"### 🖼 Image {idx+1}: {uploaded_file.name}")

#         try:
#             image = Image.open(uploaded_file).convert("RGB")  
#         except UnidentifiedImageError:
#             st.error(f"❌ Error: Unable to open {uploaded_file.name}. Skipping...")
#             continue

#         st.write("🚀 Running Object Detection...")

#         detected_objects = set()
#         cnn_image, vit_image = None, None
        
#         # Store original results before formatting
#         raw_cnn_results, raw_vit_results = [], []
        
#         # CNN Model Processing
#         if run_cnn:
#             raw_cnn_results, cnn_time, cnn_image, cnn_avg_conf, cnn_accuracy, cnn_flops = detect_objects_cnn(image.copy())
#             # Ensure accuracy is never greater than 1.0 (100%)
#             cnn_accuracy = min(cnn_accuracy, 1.0)
#             cnn_times.append(cnn_time)
#             cnn_confidences.append(cnn_avg_conf)
#             cnn_accuracies.append(cnn_accuracy)  # Store the accuracy from the model
#             cnn_flops_list.append(cnn_flops)
#             cnn_count += 1
#             cnn_results = format_results(raw_cnn_results)
#             cnn_total_objects += len(cnn_results)

#         # ViT Model Processing
#         if run_vit:
#             raw_vit_results, vit_time, vit_image, vit_avg_conf, vit_accuracy, vit_flops = detect_objects_vit(image.copy())
#             # Ensure accuracy is never greater than 1.0 (100%)
#             vit_accuracy = min(vit_accuracy, 1.0)
#             vit_times.append(vit_time)
#             vit_confidences.append(vit_avg_conf)
#             vit_accuracies.append(vit_accuracy)  # Store the accuracy from the model
#             vit_flops_list.append(vit_flops)
#             vit_count += 1
#             vit_results = format_results(raw_vit_results)
#             vit_total_objects += len(vit_results)

#         # Display images in the same row
#         col1, col2, col3 = st.columns(3)
#         col1.image(image, caption="Original Image", use_container_width=True)
#         if run_cnn and cnn_image is not None:
#             col2.image(cnn_image, caption="CNN Detection", use_container_width=True)
#         if run_vit and vit_image is not None:
#             col3.image(vit_image, caption="ViT Detection", use_container_width=True)

#         # Create dictionaries to store object-specific metrics for this image
#         image_object_confidences = {}
#         image_object_accuracies = {}
        
#         # Process CNN detections
#         if run_cnn:
#             for obj, confidence in cnn_results:
#                 detected_objects.add(obj)
#                 # Store confidence and time for each object
#                 category_results[obj]["cnn"].append((confidence, cnn_time))
#                 category_counts[obj]["cnn"] += 1
                
#                 # Store object-specific metrics
#                 if obj not in image_object_confidences:
#                     image_object_confidences[obj] = {"cnn_confidence": confidence, "vit_confidence": None}
#                 else:
#                     image_object_confidences[obj]["cnn_confidence"] = confidence
                    
#                 # FIXED: Remove randomization - use confidence to adjust accuracy instead
#                 # This ensures consistency and prevents biased results
#                 confidence_factor = max(0.8, min(confidence, 1.0))  # Scale confidence to reasonable range
#                 object_accuracy = min(cnn_accuracy * confidence_factor, 1.0)
                
#                 if obj not in image_object_accuracies:
#                     image_object_accuracies[obj] = {"cnn_accuracy": object_accuracy, "vit_accuracy": None}
#                 else:
#                     image_object_accuracies[obj]["cnn_accuracy"] = object_accuracy
                    
#         # Process ViT detections
#         if run_vit:
#             for obj, confidence in vit_results:
#                 detected_objects.add(obj)
#                 # Store confidence and time for each object
#                 category_results[obj]["vit"].append((confidence, vit_time))
#                 category_counts[obj]["vit"] += 1
                
#                 # Store object-specific confidence
#                 if obj not in image_object_confidences:
#                     image_object_confidences[obj] = {"cnn_confidence": None, "vit_confidence": confidence}
#                 else:
#                     image_object_confidences[obj]["vit_confidence"] = confidence
                    
#                 # FIXED: Remove randomization - use confidence to adjust accuracy instead
#                 # Apply same logic to ViT for fairness
#                 confidence_factor = max(0.8, min(confidence, 1.0))  # Scale confidence to reasonable range
#                 object_accuracy = min(vit_accuracy * confidence_factor, 1.0)
                
#                 if obj not in image_object_accuracies:
#                     image_object_accuracies[obj] = {"cnn_accuracy": None, "vit_accuracy": object_accuracy}
#                 else:
#                     image_object_accuracies[obj]["vit_accuracy"] = object_accuracy
                    
#         # Add image and object details to each detected category
#         for obj in detected_objects:
#             # Get object-specific metrics
#             obj_confidences = image_object_confidences.get(obj, {"cnn_confidence": None, "vit_confidence": None})
#             obj_accuracies = image_object_accuracies.get(obj, {"cnn_accuracy": None, "vit_accuracy": None})
            
#             # Store the image with its detection data
#             category_images[obj].append((uploaded_file, cnn_image, vit_image))
            
#             # Store object-specific metrics for this image - use different values for accuracy and confidence
#             category_image_details[obj].append({
#                 "image": uploaded_file.name,
#                 "cnn_accuracy": obj_accuracies["cnn_accuracy"] if run_cnn and obj_accuracies["cnn_accuracy"] is not None else None,
#                 "cnn_confidence": obj_confidences["cnn_confidence"] if run_cnn and obj_confidences["cnn_confidence"] is not None else None,
#                 "vit_accuracy": obj_accuracies["vit_accuracy"] if run_vit and obj_accuracies["vit_accuracy"] is not None else None,
#                 "vit_confidence": obj_confidences["vit_confidence"] if run_vit and obj_confidences["vit_confidence"] is not None else None
#             })

#         progress_bar.progress((idx + 1) / num_files)

#     # Sidebar to select categories
#     st.sidebar.header("Select Category")
#     category_selected = st.sidebar.selectbox("Choose a category:", list(category_images.keys()))

#     if category_selected:
#         cnn_objects = category_counts[category_selected]["cnn"]
#         vit_objects = category_counts[category_selected]["vit"]

#         st.subheader(f"📊 *Objects detected in '{category_selected}' Category:*")
#         for img, cnn_img, vit_img in category_images[category_selected]:
#             col1, col2, col3 = st.columns(3)
#             col1.image(img, caption="Original Image", use_container_width=True)
#             if run_cnn and cnn_img is not None:
#                 col2.image(cnn_img, caption="CNN Detection", use_container_width=True)
#             if run_vit and vit_img is not None:
#                 col3.image(vit_img, caption="ViT Detection", use_container_width=True)

#     # Performance Comparison Graphs
#     st.write("## 📊 Model Comparison Graphs")
    
#     # Training Loss Graph
#     fig, ax = plt.subplots()
#     ax.plot(range(1, len(cnn_losses) + 1), cnn_losses, label="CNN (ResNet-50)", marker="o", color="blue")
#     ax.plot(range(1, len(vit_losses) + 1), vit_losses, label="ViT (ViT-Base)", marker="s", color="orange")
#     ax.set_xlabel("Epochs")
#     ax.set_ylabel("Training Loss")
#     ax.set_title("Training Loss Over Time")
#     ax.legend()
#     st.pyplot(fig)

#     # Inference Time Graph
#     fig, ax = plt.subplots()
#     ax.bar(["CNN", "ViT"], [np.mean(cnn_times) if cnn_times else 0, np.mean(vit_times) if vit_times else 0], color=["blue", "orange"])
#     ax.set_ylabel("Avg Inference Time (ms)")
#     ax.set_title("Inference Time per Image")
#     st.pyplot(fig)

#     # FLOPs Graph
#     fig, ax = plt.subplots()
#     ax.bar(["CNN (ResNet-50)", "ViT (ViT-Base)"], 
#            [np.mean(cnn_flops_list) if cnn_flops_list else 0, np.mean(vit_flops_list) if vit_flops_list else 0], 
#            color=["blue", "orange"])
#     ax.set_ylabel("FLOPs (Floating Point Operations)")
#     ax.set_title("FLOPs Comparison")
#     st.pyplot(fig)
   
#     # Compute overall metrics using the correct values
#     avg_cnn_conf = np.mean(cnn_confidences) if cnn_confidences else 0
#     avg_vit_conf = np.mean(vit_confidences) if vit_confidences else 0
    
#     # Use the separate accuracy values from the models
#     # Ensure accuracy is never greater than 100%
#     avg_cnn_acc = min(np.mean(cnn_accuracies) * 100, 100.0) if cnn_accuracies else 0
#     avg_vit_acc = min(np.mean(vit_accuracies) * 100, 100.0) if vit_accuracies else 0
    
#     # Calculate category-specific metrics
#     cnn_category_confidences = []
#     vit_category_confidences = []
#     cnn_category_accuracies = []
#     vit_category_accuracies = []
    
#     # FIXED: Check for data availability before calculating category metrics
#     category_comparison_valid = True
    
#     if category_selected:
#         cnn_category_confidences = [conf for conf, _ in category_results[category_selected]["cnn"]]
#         vit_category_confidences = [conf for conf, _ in category_results[category_selected]["vit"]]
        
#         # Extract per-image accuracies for the selected category
#         for details in category_image_details[category_selected]:
#             if details["cnn_accuracy"] is not None:
#                 cnn_category_accuracies.append(details["cnn_accuracy"])
#             if details["vit_accuracy"] is not None:
#                 vit_category_accuracies.append(details["vit_accuracy"])

#         # FIXED: Check if both models have data for the selected category
#         if len(cnn_category_accuracies) == 0 or len(vit_category_accuracies) == 0:
#             category_comparison_valid = False
#             st.warning(f"⚠️ Not enough data for both models in '{category_selected}' category for a fair comparison.")

#         avg_cnn_category_conf = np.mean(cnn_category_confidences) if cnn_category_confidences else 0
#         avg_vit_category_conf = np.mean(vit_category_confidences) if vit_category_confidences else 0
        
#         # Calculate average category-specific accuracy
#         avg_cnn_category_acc = np.mean(cnn_category_accuracies) * 100 if cnn_category_accuracies else 0
#         avg_vit_category_acc = np.mean(vit_category_accuracies) * 100 if vit_category_accuracies else 0

#     else:
#         avg_cnn_category_conf = 0
#         avg_vit_category_conf = 0
#         avg_cnn_category_acc = 0
#         avg_vit_category_acc = 0

#     # Calculate inference time metrics
#     avg_cnn_time = np.mean(cnn_times) if cnn_times else 0
#     avg_vit_time = np.mean(vit_times) if vit_times else 0
    
#     # Calculate FLOPs metrics
#     avg_cnn_flops = np.mean(cnn_flops_list) if cnn_flops_list else 0
#     avg_vit_flops = np.mean(vit_flops_list) if vit_flops_list else 0
    
#     # Calculate training loss final values
#     cnn_final_loss = cnn_losses[-1] if cnn_losses else 0
#     vit_final_loss = vit_losses[-1] if vit_losses else 0

#     # FIXED: Create a more balanced scoring system
#     cnn_wins_overall = 0
#     vit_wins_overall = 0
#     tie_count_overall = 0
#     total_metrics_overall = 0  # Track the total number of metrics compared
    
#     # For category-specific metrics
#     cnn_wins_category = 0
#     vit_wins_category = 0
#     tie_count_category = 0
#     total_metrics_category = 0
    
#     # Define a function to compare metrics and update win counts
#     def compare_metric(cnn_value, vit_value, higher_is_better=True, min_threshold=0, category_specific=False):
#         global cnn_wins_overall, vit_wins_overall, tie_count_overall, total_metrics_overall
#         global cnn_wins_category, vit_wins_category, tie_count_category, total_metrics_category
        
#         # Skip comparison if values are too small or invalid
#         if cnn_value <= min_threshold and vit_value <= min_threshold:
#             return
            
#         if category_specific:
#             total_metrics_category += 1
#         else:
#             total_metrics_overall += 1
        
#         if higher_is_better:
#             # For metrics where higher values are better (accuracy, confidence)
#             if abs(cnn_value - vit_value) < 0.001:  # Consider a small difference as a tie
#                 if category_specific:
#                     tie_count_category += 1
#                 else:
#                     tie_count_overall += 1
#             elif cnn_value > vit_value:
#                 if category_specific:
#                     cnn_wins_category += 1
#                 else:
#                     cnn_wins_overall += 1
#             else:
#                 if category_specific:
#                     vit_wins_category += 1
#                 else:
#                     vit_wins_overall += 1
#         else:
#             # For metrics where lower values are better (time, FLOPs, loss)
#             if abs(cnn_value - vit_value) < 0.001:  # Consider a small difference as a tie
#                 if category_specific:
#                     tie_count_category += 1
#                 else:
#                     tie_count_overall += 1
#             elif cnn_value < vit_value:
#                 if category_specific:
#                     cnn_wins_category += 1
#                 else:
#                     cnn_wins_overall += 1
#             else:
#                 if category_specific:
#                     vit_wins_category += 1
#                 else:
#                     vit_wins_overall += 1
    
#     # Compare overall metrics
#     compare_metric(cnn_total_objects, vit_total_objects, higher_is_better=True)
#     compare_metric(avg_cnn_acc, avg_vit_acc, higher_is_better=True)
#     compare_metric(avg_cnn_conf, avg_vit_conf, higher_is_better=True)
#     compare_metric(avg_cnn_time, avg_vit_time, higher_is_better=False, min_threshold=0.01)
#     compare_metric(avg_cnn_flops, avg_vit_flops, higher_is_better=False, min_threshold=0.01)
#     compare_metric(cnn_final_loss, vit_final_loss, higher_is_better=False, min_threshold=0.01)
    
#     # Add category-specific comparisons if a category is selected and valid
#     if category_selected and category_comparison_valid:
#         # Compare category detections
#         compare_metric(cnn_objects, vit_objects, higher_is_better=True, category_specific=True)
        
#         # Compare category accuracy
#         compare_metric(avg_cnn_category_acc, avg_vit_category_acc, higher_is_better=True, category_specific=True)
        
#         # Compare category confidence
#         compare_metric(avg_cnn_category_conf, avg_vit_category_conf, higher_is_better=True, category_specific=True)
        
#         # Calculate and compare category-specific inference times if available
#         if category_results[category_selected]["cnn"] and category_results[category_selected]["vit"]:
#             category_cnn_times = [time for _, time in category_results[category_selected]["cnn"]]
#             category_vit_times = [time for _, time in category_results[category_selected]["vit"]]
#             if category_cnn_times and category_vit_times:
#                 avg_cnn_category_time = np.mean(category_cnn_times)
#                 avg_vit_category_time = np.mean(category_vit_times)
#                 compare_metric(avg_cnn_category_time, avg_vit_category_time, higher_is_better=False, category_specific=True)
    
#     # Determine the best model overall
#     if cnn_wins_overall > vit_wins_overall:
#         best_model_overall = "CNN"
#     elif vit_wins_overall > cnn_wins_overall:
#         best_model_overall = "ViT"
#     else:
#         # If there's a tie in win count, we use accuracy as primary tiebreaker
#         if avg_cnn_acc > avg_vit_acc:
#             best_model_overall = "CNN"
#         elif avg_vit_acc > avg_cnn_acc:
#             best_model_overall = "ViT"
#         else:
#             # If accuracy is also tied, use inference time as secondary tiebreaker
#             if avg_cnn_time < avg_vit_time:
#                 best_model_overall = "CNN"
#             else:
#                 best_model_overall = "ViT"
    
#     # Determine the best model for the selected category
#     best_model_category = "N/A"
#     if category_selected and category_comparison_valid:
#         if cnn_wins_category > vit_wins_category:
#             best_model_category = "CNN"
#         elif vit_wins_category > cnn_wins_category:
#             best_model_category = "ViT"
#         else:
#             # If there's a tie in win count, we use accuracy as primary tiebreaker
#             if avg_cnn_category_acc > avg_vit_category_acc:
#                 best_model_category = "CNN"
#             elif avg_vit_category_acc > avg_cnn_category_acc:
#                 best_model_category = "ViT"
#             else:
#                 # If accuracy is also tied, use confidence as secondary tiebreaker
#                 if avg_cnn_category_conf > avg_vit_category_conf:
#                     best_model_category = "CNN"
#                 else:
#                     best_model_category = "ViT"
    
#     # Add confidence level to the best model conclusion
#     confidence_margin_overall = abs(cnn_wins_overall - vit_wins_overall) / max(total_metrics_overall, 1)
#     if confidence_margin_overall > 0.5:
#         confidence_level_overall = "high confidence"
#     elif confidence_margin_overall > 0.2:
#         confidence_level_overall = "moderate confidence"
#     else:
#         confidence_level_overall = "slight edge"
    
#     # Add confidence level for category-specific conclusion
#     confidence_level_category = "N/A"
#     if category_selected and category_comparison_valid:
#         confidence_margin_category = abs(cnn_wins_category - vit_wins_category) / max(total_metrics_category, 1)
#         if confidence_margin_category > 0.5:
#             confidence_level_category = "high confidence"
#         elif confidence_margin_category > 0.2:
#             confidence_level_category = "moderate confidence"
#         else:
#             confidence_level_category = "slight edge"
    
#     st.subheader("📊 *RESULT METRICS*")
#     results_data = {
#         "Metric": [
#             "Total Objects Detected",
#             "Overall Accuracy (%)",
#             "Overall Confidence",
#             f"Objects detected in '{category_selected}' Category",
#             f"Accuracy(%) in '{category_selected}' Category ",
#             f"Confidence in '{category_selected}' Category",
#             f"Inference Time (ms) in '{category_selected}' Category",
#             f"FLOPs in '{category_selected}' Category",
#             f"Training Loss in '{category_selected}' Category",
#             "🏆 Best Model Overall"
#         ],
#         "CNN": [
#             cnn_total_objects,
#             f"{avg_cnn_acc:.2f}%",
#             f"{avg_cnn_conf:.2f}",
#             f"{cnn_objects} objects" if category_selected else "-",
#             f"{avg_cnn_category_acc:.2f}%" if category_selected else "-",
#             f"{avg_cnn_category_conf:.4f}" if category_selected else "-",  # Updated to 4 decimal places
#             f"{avg_cnn_time:.2f}" if cnn_times else "-",
#             f"{avg_cnn_flops:.2f}" if cnn_flops_list else "-",
#             f"{cnn_final_loss:.2f}" if cnn_losses else "-",
#             "✓" if best_model_overall == "CNN" else "-"
#         ],
#          "ViT": [
#             vit_total_objects,
#             f"{avg_vit_acc:.2f}%",
#             f"{avg_vit_conf:.2f}",
#             f"{vit_objects} objects" if category_selected else "-",
#             f"{avg_vit_category_acc:.2f}%" if category_selected else "-",
#             f"{avg_vit_category_conf:.4f}" if category_selected else "-",  # Updated to 4 decimal places
#             f"{avg_vit_time:.2f}" if vit_times else "-",
#             f"{avg_vit_flops:.2f}" if vit_flops_list else "-",
#             f"{vit_final_loss:.2f}" if vit_losses else "-",
#             "✓" if best_model_overall == "ViT" else "-"
#         ]
#     }
#     st.table(pd.DataFrame(results_data))

#     # Display model comparison details with the same metrics as RESULT METRICS
#     with st.expander("📊 *Detailed Model Comparison*"):
#         st.write("### Model Comparison Details")
#         st.write("The best model is determined by comparing key performance indicators:")
        
#         # Create a simple explanation of how the best model was chosen
#         comparison_text = f"""
#         ## Overall Comparison
        
#         ### Overall Metrics Summary
#         - For overall comparison, we focused on these 3 metrics:
#           1. Total Objects Detected
#           2. Overall Accuracy (%)
#           3. Overall Confidence
          
#         - CNN wins in {cnn_wins_overall} metrics
#         - ViT wins in {vit_wins_overall} metrics
        
#         ### Overall Metric-by-Metric Comparison:
#         1. **Total Objects Detected**: {'CNN' if cnn_total_objects > vit_total_objects else 'ViT' if vit_total_objects > cnn_total_objects else 'Tie'} ({cnn_total_objects} vs {vit_total_objects})
#         2. **Overall Accuracy**: {'CNN' if avg_cnn_acc > avg_vit_acc else 'ViT' if avg_vit_acc > avg_cnn_acc else 'Tie'} ({avg_cnn_acc:.2f}% vs {avg_vit_acc:.2f}%)
#         3. **Overall Confidence**: {'CNN' if avg_cnn_conf > avg_vit_conf else 'ViT' if avg_vit_conf > avg_cnn_conf else 'Tie'} ({avg_cnn_conf:.2f} vs {avg_vit_conf:.2f})
        
#         ### Best Model Overall: {best_model_overall} (with {confidence_level_overall})
#         """
        
#         # Add category-specific comparison section if a category is selected
#         if category_selected and category_comparison_valid:
#             category_comparison = f"""
#             ## Category-Specific Comparison for '{category_selected}'
            
#             ### Category Metrics Summary
#             - For category comparison, we focused on these 6 metrics:
#               1. Objects detected
#               2. Accuracy (%)
#               3. Confidence
#               4. Inference Time (ms)
#               5. FLOPs
#               6. Training Loss
              
#             - CNN wins in {cnn_wins_category} metrics
#             - ViT wins in {vit_wins_category} metrics
            
#             # ### Category Metric-by-Metric Comparison:
#             # 1. **Objects Detected**: {'CNN' if cnn_objects > vit_objects else 'ViT' if vit_objects > cnn_objects else 'Tie'} ({cnn_objects} vs {vit_objects})
#             # 2. **Accuracy**: {'CNN' if avg_cnn_category_acc > avg_vit_category_acc else 'ViT' if avg_vit_category_acc > avg_cnn_category_acc else 'Tie'} ({avg_cnn_category_acc:.2f}% vs {avg_vit_category_acc:.2f}%)
#             # 3. **Confidence**: {'CNN' if avg_cnn_category_conf > avg_vit_category_conf else 'ViT' if avg_vit_category_conf > avg_cnn_category_conf else 'Tie'} ({avg_cnn_category_conf:.4f} vs {avg_vit_category_conf:.4f})
#             # 4. **Inference Time**: {'CNN' if avg_cnn_time < avg_vit_time else 'ViT' if avg_vit_time < avg_cnn_time else 'Tie'} ({avg_cnn_time:.2f}ms vs {avg_vit_time:.2f}ms) - lower is better
#             # 5. **Computational Efficiency (FLOPs)**: {'CNN' if avg_cnn_flops < avg_vit_flops else 'ViT' if avg_vit_flops < avg_cnn_flops else 'Tie'} ({avg_cnn_flops:.2f} vs {avg_vit_flops:.2f}) - lower is better
#             # 6. **Training Loss**: {'CNN' if cnn_final_loss < vit_final_loss else 'ViT' if vit_final_loss < cnn_final_loss else 'Tie'} ({cnn_final_loss:.2f} vs {vit_final_loss:.2f}) - lower is better
            
#             ### Best Model for {category_selected}: {best_model_category} (with {confidence_level_category})
#             """
#             comparison_text += category_comparison
#         else:
#             comparison_text += "\n\n## Category-Specific Comparison\n\nNo category selected or not enough data for comparison."
            
#         st.write(comparison_text)
        
#         st.write(f"### Final Result")
#         st.write(f"Based on winning in {cnn_wins_overall if best_model_overall == 'CNN' else vit_wins_overall} out of 3 key performance metrics, *{best_model_overall}* is the better model overall with {confidence_level_overall}.")
        
#         if category_selected and category_comparison_valid:
#             st.write(f"For the category '{category_selected}', *{best_model_category}* is the better model with {confidence_level_category}, based on winning in {cnn_wins_category if best_model_category == 'CNN' else vit_wins_category} out of 6 key category-specific metrics.")

#     # Display Accuracy and Confidence
#     if category_selected:
#         st.subheader(f"📊 *Accuracy & Confidence per Image in '{category_selected}' Category*")
    
#         # Only display details for images where this category was actually detected
#         filtered_data = [
#             {
#                 "Image": details["image"],
#                 "CNN Accuracy (%)": f"{details['cnn_accuracy'] * 100:.4f}%" if details["cnn_accuracy"] is not None else "-",
#                 "CNN Confidence": f"{details['cnn_confidence']:.4f}" if details["cnn_confidence"] is not None else "-",
#                 "ViT Accuracy (%)": f"{details['vit_accuracy'] * 100:.4f}%" if details["vit_accuracy"] is not None else "-",
#                 "ViT Confidence": f"{details['vit_confidence']:.4f}" if details["vit_confidence"] is not None else "-"
#             }
#             for details in category_image_details[category_selected]
#         ]

#         # Convert to Pandas DataFrame and display
#         if filtered_data:
#             st.table(pd.DataFrame(filtered_data))
#         else:
#             st.write("⚠ No data available for this category.")
             
#neww

# import streamlit as st
# from PIL import Image, UnidentifiedImageError
# import sys
# import os
# import time
# from collections import defaultdict
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd

# # Set Streamlit page configuration
# st.set_page_config(page_title="Object Detection: CNN vs ViT", layout="wide")

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# from coco_labels import load_coco_labels
# from models.cnn_model import detect_objects_cnn, get_cnn_training_loss
# from models.vit_model import detect_objects_vit, get_vit_training_loss

# # Title
# st.title("🔍 Object Detection: CNN vs ViT")

# # Sidebar options
# st.sidebar.header("Settings")
# run_cnn = st.sidebar.checkbox("Run CNN Model", value=True)
# run_vit = st.sidebar.checkbox("Run ViT Model", value=True)

# # Multi-file uploader
# uploaded_files = st.file_uploader("📂 Upload up to 100 images", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

# if uploaded_files:
#     num_files = len(uploaded_files)
#     st.write(f"✅ Uploaded {num_files} image(s).")

#     if num_files > 100:
#         st.warning("⚠ You uploaded more than 100 images. Only the first 100 will be processed.")
#         uploaded_files = uploaded_files[:100]

#     # Load COCO labels
#     coco_labels = load_coco_labels("dataset/annotations/instances_train2017.json")
#     coco_labels.update(load_coco_labels("dataset/annotations/instances_val2017.json"))

#     # Helper function to format detection results
#     def format_results(results):
#         formatted_results = []
#         for item in results:
#             obj_id = item.get("Object") or item.get("Label")  
#             obj_label = coco_labels.get(obj_id, f"Unknown ({obj_id})")
#             confidence = item.get("Image confidence", 0)
#             formatted_results.append((obj_label, confidence))  
#         return formatted_results

#     # Categorization storage
#     category_images = defaultdict(list)
#     category_results = defaultdict(lambda: {"cnn": [], "vit": []})
#     category_image_details = defaultdict(list)
    
#     # Performance tracking
#     cnn_times, vit_times = [], []
#     cnn_flops_list, vit_flops_list = [], []
#     cnn_losses, vit_losses = get_cnn_training_loss(), get_vit_training_loss()
    
#     # For tracking overall metrics
#     cnn_confidences, vit_confidences = [], []
#     cnn_accuracies, vit_accuracies = [], []  # Lists for tracking model accuracies
    
#     cnn_count, vit_count = 0, 0
#     cnn_total_objects, vit_total_objects = 0, 0  
#     category_counts = defaultdict(lambda: {"cnn": 0, "vit": 0})
    
#     # Category-specific performance tracking
#     category_cnn_times = defaultdict(list)
#     category_vit_times = defaultdict(list)
#     category_cnn_flops = defaultdict(list)
#     category_vit_flops = defaultdict(list)

#     # Progress bar
#     progress_bar = st.progress(0)

#     # Process each image
#     for idx, uploaded_file in enumerate(uploaded_files):
#         st.write(f"### 🖼 Image {idx+1}: {uploaded_file.name}")

#         try:
#             image = Image.open(uploaded_file).convert("RGB")  
#         except UnidentifiedImageError:
#             st.error(f"❌ Error: Unable to open {uploaded_file.name}. Skipping...")
#             continue

#         st.write("🚀 Running Object Detection...")

#         detected_objects = set()
#         cnn_image, vit_image = None, None
        
#         # Store original results before formatting
#         raw_cnn_results, raw_vit_results = [], []
#         image_object_confidences = {}
#         # CNN Model Processing
#         if run_cnn:
#             raw_cnn_results, cnn_time, cnn_image, cnn_avg_conf, cnn_accuracy, cnn_flops = detect_objects_cnn(image.copy())
#             cnn_accuracy = min(cnn_accuracy, 1.0)
#             cnn_times.append(cnn_time)
#             cnn_confidences.append(cnn_avg_conf)
#             cnn_accuracies.append(cnn_accuracy)
#             cnn_flops_list.append(cnn_flops)
#             cnn_count += 1
#             cnn_results = format_results(raw_cnn_results)
#             cnn_total_objects += len(cnn_results)

#         # ViT Model Processing
#         if run_vit:
#             raw_vit_results, vit_time, vit_image, vit_avg_conf, vit_accuracy, vit_flops = detect_objects_vit(image.copy())
#             vit_accuracy = min(vit_accuracy, 1.0)
#             vit_times.append(vit_time)
#             vit_confidences.append(vit_avg_conf)
#             vit_accuracies.append(vit_accuracy)
#             vit_flops_list.append(vit_flops)
#             vit_count += 1
#             vit_results = format_results(raw_vit_results)
#             vit_total_objects += len(vit_results)

#         # Display images in the same row
#         col1, col2, col3 = st.columns(3)
#         col1.image(image, caption="Original Image", use_container_width=True)
#         if run_cnn and cnn_image is not None:
#             col2.image(cnn_image, caption="CNN Detection", use_container_width=True)
#         if run_vit and vit_image is not None:
#             col3.image(vit_image, caption="ViT Detection", use_container_width=True)
        
#         # Process CNN detections
#         if run_cnn:
#             for obj, confidence in cnn_results:
#                 detected_objects.add(obj)
#                 category_results[obj]["cnn"].append((confidence, cnn_time))
#                 category_counts[obj]["cnn"] += 1
#                 category_cnn_times[obj].append(cnn_time)
#                 category_cnn_flops[obj].append(cnn_flops)
                
#                 # Store confidence for this object in this image
#                 if obj not in image_object_confidences:
#                     image_object_confidences[obj] = {"cnn_confidence": confidence, "vit_confidence": None}
#                 else:
#                     image_object_confidences[obj]["cnn_confidence"] = confidence

#         # Process ViT detections
#         if run_vit:
#             for obj, confidence in vit_results:
#                 detected_objects.add(obj)
#                 category_results[obj]["vit"].append((confidence, vit_time))
#                 category_counts[obj]["vit"] += 1
#                 category_vit_times[obj].append(vit_time)
#                 category_vit_flops[obj].append(vit_flops)
                
#                 # Store confidence for this object in this image
#                 if obj not in image_object_confidences:
#                     image_object_confidences[obj] = {"cnn_confidence": None, "vit_confidence": confidence}
#                 else:
#                     image_object_confidences[obj]["vit_confidence"] = confidence

#         # Add image and object details to each detected category
#         for obj in detected_objects:
#             category_images[obj].append((uploaded_file, cnn_image, vit_image))
#             category_image_details[obj].append({
#                 "image": uploaded_file.name,
#                 "cnn_accuracy": cnn_accuracy if run_cnn else None,
#                 "cnn_confidence": image_object_confidences[obj]["cnn_confidence"] if run_cnn else None,
#                 "vit_accuracy": vit_accuracy if run_vit else None,
#                 "vit_confidence": image_object_confidences[obj]["vit_confidence"] if run_vit else None
#             })

#         progress_bar.progress((idx + 1) / num_files)

#     # Sidebar to select categories
#     st.sidebar.header("Select Category")
#     category_selected = st.sidebar.selectbox("Choose a category:", list(category_images.keys()))

#     if category_selected:
#         cnn_objects = category_counts[category_selected]["cnn"]
#         vit_objects = category_counts[category_selected]["vit"]

#         st.subheader(f"📊 *Objects detected in '{category_selected}' Category:*")
#         for img, cnn_img, vit_img in category_images[category_selected]:
#             col1, col2, col3 = st.columns(3)
#             col1.image(img, caption="Original Image", use_container_width=True)
#             if run_cnn and cnn_img is not None:
#                 col2.image(cnn_img, caption="CNN Detection", use_container_width=True)
#             if run_vit and vit_img is not None:
#                 col3.image(vit_img, caption="ViT Detection", use_container_width=True)

#     # Performance Comparison Graphs
#     st.write("## 📊 Model Comparison Graphs")
#     def create_training_loss_graph(cnn_loss, vit_loss, title_prefix=""):
#         fig = plt.figure(figsize=(8, 5))
#         plt.plot(range(1, len(cnn_loss) + 1), cnn_loss, label="CNN (ResNet-50)", marker="o", color="blue")
#         plt.plot(range(1, len(vit_loss) + 1), vit_loss, label="ViT (ViT-Base)", marker="s", color="orange")
#         plt.xlabel("Epochs")
#         plt.ylabel("Training Loss")
#         plt.title(f"{title_prefix}Training Loss Over Time")
#         plt.legend()
#         plt.grid(alpha=0.3)
#         return fig
#     # Function to create performance graphs
#     def create_performance_graphs(cnn_times_data, vit_times_data, cnn_flops_data, vit_flops_data, cnn_loss, vit_loss, title_prefix=""):
#         # Training Loss Graph
#         fig_loss = plt.figure(figsize=(8, 5))
#         plt.plot(range(1, len(cnn_loss) + 1), cnn_loss, label="CNN (ResNet-50)", marker="o", color="blue")
#         plt.plot(range(1, len(vit_loss) + 1), vit_loss, label="ViT (ViT-Base)", marker="s", color="orange")
#         plt.xlabel("Epochs")
#         plt.ylabel("Training Loss")
#         plt.title(f"{title_prefix}Training Loss Over Time")
#         plt.legend()
#         plt.grid(alpha=0.3)
        
#         # Inference Time Graph
#         fig_time = plt.figure(figsize=(8, 5))
#         plt.bar(["CNN", "ViT"], 
#                [np.mean(cnn_times_data) if cnn_times_data else 0, 
#                 np.mean(vit_times_data) if vit_times_data else 0], 
#                color=["blue", "orange"])
#         plt.ylabel("Avg Inference Time (ms)")
#         plt.title(f"{title_prefix}Inference Time per Image")
#         plt.grid(axis='y', alpha=0.3)
        
#         # FLOPs Graph
#         fig_flops = plt.figure(figsize=(8, 5))
#         plt.bar(["CNN (ResNet-50)", "ViT (ViT-Base)"], 
#                [np.sum(cnn_flops_data) if cnn_flops_data else 0, 
#                 np.sum(vit_flops_data) if vit_flops_data else 0], 
#                color=["blue", "orange"])
#         plt.ylabel("Total FLOPs (Floating Point Operations)")
#         plt.title(f"{title_prefix}FLOPs Comparison")
#         plt.grid(axis='y', alpha=0.3)
        
#         return fig_loss, fig_time, fig_flops

#     # Function to create category-specific performance graphs (without training loss)
#     def create_category_performance_graphs(cnn_times_data, vit_times_data, cnn_flops_data, vit_flops_data, title_prefix=""):
#         # Inference Time Graph
#         fig_time = plt.figure(figsize=(8, 5))
#         plt.bar(["CNN", "ViT"], 
#                [np.mean(cnn_times_data) if cnn_times_data else 0, 
#                 np.mean(vit_times_data) if vit_times_data else 0], 
#                color=["blue", "orange"])
#         plt.ylabel("Avg Inference Time (ms)")
#         plt.title(f"{title_prefix}Inference Time per Image")
#         plt.grid(axis='y', alpha=0.3)
        
#         # FLOPs Graph
#         fig_flops = plt.figure(figsize=(8, 5))
#         plt.bar(["CNN (ResNet-50)", "ViT (ViT-Base)"], 
#                [np.sum(cnn_flops_data) if cnn_flops_data else 0, 
#                 np.sum(vit_flops_data) if vit_flops_data else 0], 
#                color=["blue", "orange"])
#         plt.ylabel("Total FLOPs (Floating Point Operations)")
#         plt.title(f"{title_prefix}FLOPs Comparison")
#         plt.grid(axis='y', alpha=0.3)
        
#         return fig_time, fig_flops

#     # Create the graphs for overall data
#     overall_loss_fig, overall_time_fig, overall_flops_fig = create_performance_graphs(
#         cnn_times, vit_times, cnn_flops_list, vit_flops_list, cnn_losses, vit_losses, "Overall: "
#     )

#     # Check if we have category data
#     has_category_data = category_selected and (category_cnn_times[category_selected] or category_vit_times[category_selected])
    
#     # If we have category data, create category graphs
#     if has_category_data:
#         category_time_fig, category_flops_fig = create_category_performance_graphs(
#             category_cnn_times[category_selected], 
#             category_vit_times[category_selected], 
#             category_cnn_flops[category_selected], 
#             category_vit_flops[category_selected], 
#             f"'{category_selected}' Category: "
#         )
    
#     st.write("### Training Loss Comparison")
#     col1, col2 = st.columns(2)

#     # Overall Training Loss Graph
#     with col1:
#         overall_loss_fig = create_training_loss_graph(cnn_losses, vit_losses, "Overall: ")
#         st.pyplot(overall_loss_fig)

#     # Category-wise Training Loss Graph
#     with col2:
#         if category_selected and has_category_data:
#             # Use the same training loss data for category-wise (since training loss is model-specific, not category-specific)
#             category_loss_fig = create_training_loss_graph(cnn_losses, vit_losses, f"'{category_selected}' Category: ")
#             st.pyplot(category_loss_fig)
#         else:
#             st.write("No category-specific data available.")

#     # Row 2: Inference Time Comparison
#     st.write("### Inference Time Comparison")
#     col1, col2 = st.columns(2)
#     col1.pyplot(overall_time_fig)
#     if has_category_data:
#         col2.pyplot(category_time_fig)
#     else:
#         col2.write("No category-specific data available.")

#     # Row 3: FLOPs Comparison
#     st.write("### Computational Efficiency (FLOPs) Comparison")
#     col1, col2 = st.columns(2)
#     col1.pyplot(overall_flops_fig)
#     if has_category_data:
#         col2.pyplot(category_flops_fig)
#     else:
#         col2.write("No category-specific data available.")

#     # Compute overall metrics using the correct values
#     avg_cnn_conf = np.mean(cnn_confidences) if cnn_confidences else 0
#     avg_vit_conf = np.mean(vit_confidences) if vit_confidences else 0
    
#     # Use the separate accuracy values from the models
#     # Ensure accuracy is never greater than 100%
#     avg_cnn_acc = min(np.mean(cnn_accuracies) * 100, 100.0) if cnn_accuracies else 0
#     avg_vit_acc = min(np.mean(vit_accuracies) * 100, 100.0) if vit_accuracies else 0
    
#     # Calculate category-specific metrics
#     cnn_category_confidences = []
#     vit_category_confidences = []
#     cnn_category_accuracies = []
#     vit_category_accuracies = []
    
#     # FIXED: Check for data availability before calculating category metrics
#     category_comparison_valid = True
    
#     if category_selected:
#         cnn_category_confidences = [conf for conf, _ in category_results[category_selected]["cnn"]]
#         vit_category_confidences = [conf for conf, _ in category_results[category_selected]["vit"]]
        
#         # Extract per-image accuracies for the selected category
#         for details in category_image_details[category_selected]:
#             if details["cnn_confidence"] is not None:
#                 cnn_category_confidences.append(details["cnn_confidence"])
#             if details["vit_confidence"] is not None:
#                 vit_category_confidences.append(details["vit_confidence"])

#         # FIXED: Check if both models have data for the selected category
#         if len(cnn_category_confidences) == 0 or len(vit_category_confidences) == 0:
#             category_comparison_valid = False
#             st.warning(f"⚠️ Not enough data for both models in '{category_selected}' category for a fair comparison.")

#         avg_cnn_category_conf = np.mean(cnn_category_confidences) if cnn_category_confidences else 0
#         avg_vit_category_conf = np.mean(vit_category_confidences) if vit_category_confidences else 0
        
#         # Calculate average category-specific accuracy
#         avg_cnn_category_acc = np.mean(cnn_category_accuracies) * 100 if cnn_category_accuracies else 0
#         avg_vit_category_acc = np.mean(vit_category_accuracies) * 100 if vit_category_accuracies else 0

#     else:
#         avg_cnn_category_conf = 0
#         avg_vit_category_conf = 0
#         avg_cnn_category_acc = 0
#         avg_vit_category_acc = 0

#     # Calculate inference time metrics
#     avg_cnn_time = np.mean(cnn_times) if cnn_times else 0
#     avg_vit_time = np.mean(vit_times) if vit_times else 0
    
#     # Calculate FLOPs metrics
#     avg_cnn_flops = np.mean(cnn_flops_list) if cnn_flops_list else 0
#     avg_vit_flops = np.mean(vit_flops_list) if vit_flops_list else 0
    
#     # Calculate training loss final values
#     cnn_final_loss = cnn_losses[-1] if cnn_losses else 0
#     vit_final_loss = vit_losses[-1] if vit_losses else 0

#     # UPDATED: Create a scoring system specifically focusing on the requested metrics
#     # For overall comparison, focus on: Total Objects Detected, Overall Accuracy, Overall Confidence
#     cnn_wins_overall = 0
#     vit_wins_overall = 0
    
#     # For category comparison, focus on: Objects detected, Accuracy, Confidence, Inference Time, FLOPs, Training Loss
#     cnn_wins_category = 0
#     vit_wins_category = 0
    
#     # Overall comparison - only these three metrics as requested
#     # Compare total objects detected
#     if cnn_total_objects > vit_total_objects:
#         cnn_wins_overall += 1
#     elif vit_total_objects > cnn_total_objects:
#         vit_wins_overall += 1
        
#     # Compare overall accuracy
#     if avg_cnn_acc > avg_vit_acc:
#         cnn_wins_overall += 1
#     elif avg_vit_acc > avg_cnn_acc:
#         vit_wins_overall += 1
        
#     # Compare overall confidence
#     if avg_cnn_conf > avg_vit_conf:
#         cnn_wins_overall += 1
#     elif avg_vit_conf > avg_cnn_conf:
#         vit_wins_overall += 1
    
#     # Category-specific comparison - all six metrics as requested
#     if category_selected and category_comparison_valid:
#         # Objects detected in category
#         if cnn_objects > vit_objects:
#             cnn_wins_category += 1
#         elif vit_objects > cnn_objects:
#             vit_wins_category += 1
            
#         # Accuracy in category
#         if avg_cnn_category_acc > avg_vit_category_acc:
#             cnn_wins_category += 1
#         elif avg_vit_category_acc > avg_cnn_category_acc:
#             vit_wins_category += 1
            
#         # Confidence in category
#         if avg_cnn_category_conf > avg_vit_category_conf:
#             cnn_wins_category += 1
#         elif avg_vit_category_conf > avg_cnn_category_conf:
#             vit_wins_category += 1
            
#         # Inference Time in category (lower is better)
#         if avg_cnn_time < avg_vit_time:
#             cnn_wins_category += 1
#         elif avg_vit_time < avg_cnn_time:
#             vit_wins_category += 1
            
#         # FLOPs in category (lower is better)
#         if avg_cnn_flops < avg_vit_flops:
#             cnn_wins_category += 1
#         elif avg_vit_flops < avg_cnn_flops:
#             vit_wins_category += 1
            
#         # Training Loss in category (lower is better)
#         if cnn_final_loss < vit_final_loss:
#             cnn_wins_category += 1
#         elif vit_final_loss < cnn_final_loss:
#             vit_wins_category += 1
    
#     # Determine the best model overall based on the 3 key metrics
#     if cnn_wins_overall > vit_wins_overall:
#         best_model_overall = "CNN"
#     elif vit_wins_overall > cnn_wins_overall:
#         best_model_overall = "ViT"
#     else:
#         # If there's a tie, prioritize accuracy as the tiebreaker
#         best_model_overall = "CNN" if avg_cnn_acc >= avg_vit_acc else "ViT"
    
#     # Determine the best model for the selected category based on the 6 key metrics
#     best_model_category = "N/A"
#     if category_selected and category_comparison_valid:
#         if cnn_wins_category > vit_wins_category:
#             best_model_category = "CNN"
#         elif vit_wins_category > cnn_wins_category:
#             best_model_category = "ViT"
#         else:
#             # If there's a tie, prioritize accuracy as the primary tiebreaker
#             if avg_cnn_category_acc > avg_vit_category_acc:
#                 best_model_category = "CNN"
#             elif avg_vit_category_acc > avg_cnn_category_acc:
#                 best_model_category = "ViT"
#             else:
#                 # If accuracy is tied, use confidence as the secondary tiebreaker
#                 best_model_category = "CNN" if avg_cnn_category_conf >= avg_vit_category_conf else "ViT"
    
#     # Calculate confidence levels for display purposes
#     # For overall comparison (out of 3 metrics)
#     if abs(cnn_wins_overall - vit_wins_overall) == 3:
#         confidence_level_overall = "high confidence"
#     elif abs(cnn_wins_overall - vit_wins_overall) == 2:
#         confidence_level_overall = "moderate confidence"
#     else:
#         confidence_level_overall = "slight edge"
    
#     # For category comparison (out of 6 metrics)
#     confidence_level_category = "N/A"
#     if category_selected and category_comparison_valid:
#         win_diff = abs(cnn_wins_category - vit_wins_category)
#         if win_diff >= 4:  # 4+ wins out of 6 metrics
#             confidence_level_category = "high confidence"
#         elif win_diff >= 2:  # 2-3 wins out of 6 metrics
#             confidence_level_category = "moderate confidence"
#         else:  # 0-1 win difference
#             confidence_level_category = "slight edge"

#     st.subheader("📊 *RESULT METRICS*")
#     results_data = {
#         "Metric": [
#             "Total Objects Detected",
#             "Overall Accuracy (%)",
#             "Overall Confidence",
#             f"Objects detected in '{category_selected}' Category",
#             f"Accuracy(%) in '{category_selected}' Category ",
#             f"Confidence in '{category_selected}' Category",
    #         f"Inference Time (ms) in '{category_selected}' Category",
    #         f"FLOPs in '{category_selected}' Category",
    #         f"Training Loss in '{category_selected}' Category",
    #         "🏆 Best Model Overall"
    #     ],
    #     "CNN": [
    #         cnn_total_objects,
    #         f"{avg_cnn_acc:.2f}%",
    #         f"{avg_cnn_conf:.2f}",
    #         f"{cnn_objects} objects" if category_selected else "-",
    #         f"{avg_cnn_category_acc:.2f}%" if category_selected else "-",
    #         f"{avg_cnn_category_conf:.4f}" if category_selected else "-",  # Updated to 4 decimal places
    #         f"{avg_cnn_time:.2f}" if cnn_times else "-",
    #         f"{avg_cnn_flops:.2f}" if cnn_flops_list else "-",
    #         f"{cnn_final_loss:.2f}" if cnn_losses else "-",
    #         "✓" if best_model_overall == "CNN" else "-"
    #     ],
    #      "ViT": [
    #         vit_total_objects,
    #         f"{avg_vit_acc:.2f}%",
    #         f"{avg_vit_conf:.2f}",
    #         f"{vit_objects} objects" if category_selected else "-",
    #         f"{avg_vit_category_acc:.2f}%" if category_selected else "-",
    #         f"{avg_vit_category_conf:.4f}" if category_selected else "-",  # Updated to 4 decimal places
    #         f"{avg_vit_time:.2f}" if vit_times else "-",
    #         f"{avg_vit_flops:.2f}" if vit_flops_list else "-",
    #         f"{vit_final_loss:.2f}" if vit_losses else "-",
    #         "✓" if best_model_overall == "ViT" else "-"
    #     ]
    # }
    # st.table(pd.DataFrame(results_data))

    # # Display model comparison details with the same metrics as RESULT METRICS
    # with st.expander("📊 *Detailed Model Comparison*"):
    #     st.write("### Model Comparison Details")
    #     st.write("The best model is determined by comparing key performance indicators:")
        
    #     # Create a simple explanation of how the best model was chosen
    #     comparison_text = f"""
    #     ## Overall Comparison
        
    #     ### Overall Metrics Summary
    #     - For overall comparison, we focused on these 3 metrics:
    #       1. Total Objects Detected
    #       2. Overall Accuracy (%)
    #       3. Overall Confidence
          
    #     - CNN wins in {cnn_wins_overall} metrics
    #     - ViT wins in {vit_wins_overall} metrics
        
    #     ### Overall Metric-by-Metric Comparison:
    #     1. **Total Objects Detected**: {'CNN' if cnn_total_objects > vit_total_objects else 'ViT' if vit_total_objects > cnn_total_objects else 'Tie'} ({cnn_total_objects} vs {vit_total_objects})
    #     2. **Overall Accuracy**: {'CNN' if avg_cnn_acc > avg_vit_acc else 'ViT' if avg_vit_acc > avg_cnn_acc else 'Tie'} ({avg_cnn_acc:.2f}% vs {avg_vit_acc:.2f}%)
    #     3. **Overall Confidence**: {'CNN' if avg_cnn_conf > avg_vit_conf else 'ViT' if avg_vit_conf > avg_cnn_conf else 'Tie'} ({avg_cnn_conf:.2f} vs {avg_vit_conf:.2f})
        
    #     ### Best Model Overall: {best_model_overall} (with {confidence_level_overall})
    #     """
        
    #     # Add category-specific comparison section if a category is selected
    #     if category_selected and category_comparison_valid:
    #         category_comparison = f"""
    #         ## Category-Specific Comparison for '{category_selected}'
            
    #         ### Category Metrics Summary
    #         - For category comparison, we focused on these 6 metrics:
    #           1. Objects detected
    #           2. Accuracy (%)
    #           3. Confidence
    #           4. Inference Time (ms)
    #           5. FLOPs
    #           6. Training Loss
              
    #         - CNN wins in {cnn_wins_category} metrics
    #         - ViT wins in {vit_wins_category} metrics
            
    #         ### Category Metric-by-Metric Comparison:
    #         1. **Objects Detected**: {'CNN' if cnn_objects > vit_objects else 'ViT' if vit_objects > cnn_objects else 'Tie'} ({cnn_objects} vs {vit_objects})
    #         2. **Accuracy**: {'CNN' if avg_cnn_category_acc > avg_vit_category_acc else 'ViT' if avg_vit_category_acc > avg_cnn_category_acc else 'Tie'} ({avg_cnn_category_acc:.2f}% vs {avg_vit_category_acc:.2f}%)
    #         3. **Confidence**: {'CNN' if avg_cnn_category_conf > avg_vit_category_conf else 'ViT' if avg_vit_category_conf > avg_cnn_category_conf else 'Tie'} ({avg_cnn_category_conf:.4f} vs {avg_vit_category_conf:.4f})
    #         4. **Inference Time**: {'CNN' if avg_cnn_time < avg_vit_time else 'ViT' if avg_vit_time < avg_cnn_time else 'Tie'} ({avg_cnn_time:.2f}ms vs {avg_vit_time:.2f}ms) - lower is better
    #         5. **Computational Efficiency (FLOPs)**: {'CNN' if avg_cnn_flops < avg_vit_flops else 'ViT' if avg_vit_flops < avg_cnn_flops else 'Tie'} ({avg_cnn_flops:.2f} vs {avg_vit_flops:.2f}) - lower is better
    #         6. **Training Loss**: {'CNN' if cnn_final_loss < vit_final_loss else 'ViT' if vit_final_loss < cnn_final_loss else 'Tie'} ({cnn_final_loss:.2f} vs {vit_final_loss:.2f}) - lower is better
            
    #         ### Best Model for {category_selected}: {best_model_category} (with {confidence_level_category})
    #         """
    #         comparison_text += category_comparison
    #     else:
    #         comparison_text += "\n\n## Category-Specific Comparison\n\nNo category selected or not enough data for comparison."
            
    #     st.write(comparison_text)
        
    #     st.write(f"### Final Result")
    #     st.write(f"Based on winning in {cnn_wins_overall if best_model_overall == 'CNN' else vit_wins_overall} out of 3 key performance metrics, *{best_model_overall}* is the better model overall with {confidence_level_overall}.")
        
    #     if category_selected and category_comparison_valid:
    #         st.write(f"For the category '{category_selected}', *{best_model_category}* is the better model with {confidence_level_category}, based on winning in {cnn_wins_category if best_model_category == 'CNN' else vit_wins_category} out of 6 key category-specific metrics.")

    # # Display Accuracy and Confidence
    # if category_selected:
    #     st.subheader(f"📊 *Accuracy & Confidence per Image in '{category_selected}' Category*")
    
    #     # Only display details for images where this category was actually detected
    #     filtered_data = [
    #         {
    #             "Image": details["image"],
    #             "CNN Accuracy (%)": f"{details['cnn_accuracy'] * 100:.4f}%" if details["cnn_accuracy"] is not None else "-",
    #             "CNN Confidence": f"{details['cnn_confidence']:.4f}" if details["cnn_confidence"] is not None else "-",
    #             "ViT Accuracy (%)": f"{details['vit_accuracy'] * 100:.4f}%" if details["vit_accuracy"] is not None else "-",
    #             "ViT Confidence": f"{details['vit_confidence']:.4f}" if details["vit_confidence"] is not None else "-"
    #         }
    #         for details in category_image_details[category_selected]
    #     ]

    #     # Convert to Pandas DataFrame and display
    #     if filtered_data:
    #         st.table(pd.DataFrame(filtered_data))
    #     else:
    #         st.write("⚠ No data available for this category.")