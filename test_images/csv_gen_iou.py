import os
import json
import csv
from ultralytics import YOLO
from PIL import Image, ImageDraw
import numpy as np

# --- CONFIG ---
weights_path = "/home/shaurya/Downloads/yolov8n_pokemon_more_scenes/kaggle/working/runs/detect/yolov8n_pokemon/weights/best.pt"
images_dir = "/home/shaurya/Pokemon_HACK/test_images/the-poke-war-hackathon-ai-guild-recuritment-hack/test_images"
json_file = "pokemon_kill_orders_spacy_v2.json"      # mapping file
output_csv = "pokemon_trained_longformer_nms_refined.csv"
annotated_dir = "refined_centers_nms"  # folder where annotated images will be saved
iou_threshold = 0.5  # --- NEW --- IoU threshold for Non-Maximum Suppression

os.makedirs(annotated_dir, exist_ok=True)

model = YOLO(weights_path)

# Load JSON mapping (dict: image_name -> pokemon)
with open(json_file, "r") as f:
    mappings = json.load(f)

# --- NEW --- Helper function to calculate Intersection over Union
def calculate_iou(box1, box2):
    """Calculates IoU for two bounding boxes [x1, y1, x2, y2]."""
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    if inter_area == 0:
        return 0.0

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area

# --- NEW --- Non-Maximum Suppression function
def non_max_suppression(boxes, scores, iou_thresh):
    """Performs NMS on a list of boxes and scores."""
    # Sort boxes by score in descending order
    order = np.argsort(scores)[::-1]
    
    keep = []
    while order.size > 0:
        # The index of the box with the highest score
        i = order[0]
        keep.append(i)
        
        # Get coordinates of the best box
        best_box = boxes[i]
        
        # Get coordinates of the remaining boxes
        remaining_indices = order[1:]
        remaining_boxes = [boxes[j] for j in remaining_indices]
        
        # Calculate IoU between the best box and all remaining boxes
        ious = np.array([calculate_iou(best_box, rem_box) for rem_box in remaining_boxes])
        
        # Keep only boxes with IoU less than the threshold
        low_iou_indices = np.where(ious < iou_thresh)[0]
        
        # Update the order to include only the low IoU boxes for the next iteration
        order = remaining_indices[low_iou_indices]
        
    return keep


rows = []

for image_name, target_pokemon in mappings.items():
    image_path = os.path.join(images_dir, image_name)
    if not os.path.exists(image_path):
        print(f"Skipping missing image: {image_name}")
        continue

    # --- First pass YOLO ---
    results = model(image_path, conf=0.4, verbose=False)

    centers = []
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # --- NEW --- Collect all candidate boxes for the target pokemon first
    candidate_boxes = []
    candidate_scores = []
    
    for r in results:
        names = r.names
        for box in r.boxes:
            cls_id = int(box.cls[0].item())
            cls_name = names[cls_id]
            if cls_name.lower() == target_pokemon.lower():
                candidate_boxes.append(box.xyxy[0].tolist())
                candidate_scores.append(box.conf[0].item())

    # --- NEW --- Apply NMS if candidates were found
    if candidate_boxes:
        final_indices = non_max_suppression(candidate_boxes, candidate_scores, iou_threshold)
        final_boxes = [candidate_boxes[i] for i in final_indices]
    else:
        final_boxes = []

    # --- MODIFIED --- Loop through the filtered (NMS) boxes
    for box_coords in final_boxes:
        x1, y1, x2, y2 = map(int, box_coords)

        # Crop region of interest (add some padding to be safe)
        pad = 5
        cropped = img.crop((x1 - pad, y1 - pad, x2 + pad, y2 + pad))

        # --- Second pass YOLO on cropped image ---
        second_results = model(cropped, conf=0.85, verbose=False)

        refined_center = None
        highest_conf = 0.0 # Track the highest confidence detection in the crop

        for sr in second_results:
            for sbox in sr.boxes:
                s_cls_id = int(sbox.cls[0].item())
                s_cls_name = sr.names[s_cls_id]
                s_conf = sbox.conf[0].item()

                if s_cls_name.lower() == target_pokemon.lower() and s_conf > highest_conf:
                    highest_conf = s_conf
                    sx1, sy1, sx2, sy2 = sbox.xyxy[0].tolist()

                    # Convert to original coords
                    sx1 += (x1 - pad)
                    sx2 += (x1 - pad)
                    sy1 += (y1 - pad)
                    sy2 += (y1 - pad)

                    cx = round((sx1 + sx2) / 2, 2)
                    cy = round((sy1 + sy2) / 2, 2)
                    refined_center = [cx, cy]

        # If second pass didn't detect, fall back to the NMS-filtered box center
        if refined_center:
            centers.append(refined_center)
        else:
            cx = round((x1 + x2) / 2, 2)
            cy = round((y1 + y2) / 2, 2)
            refined_center = [cx, cy]
            centers.append(refined_center)

        # --- Annotate center on the image ---
        cx, cy = refined_center
        r_size = 5  # radius for center marker
        draw.ellipse((cx - r_size, cy - r_size, cx + r_size, cy + r_size), fill="red", outline="red")
        draw.text((cx + 6, cy - 6), target_pokemon, fill="red")

    rows.append([image_name, json.dumps(centers)])

    # Save annotated image
    save_path = os.path.join(annotated_dir, image_name)
    img.save(save_path)

# Write CSV
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image_id", "points"])
    writer.writerows(rows)

print(f"✅ CSV saved to {output_csv}")
print(f"✅ Annotated images saved in {annotated_dir}/")
