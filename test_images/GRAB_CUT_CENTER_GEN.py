import os
import json
import csv
from ultralytics import YOLO
from PIL import Image, ImageDraw
import numpy as np
import cv2

# --- CONFIG ---
# Ensure these paths are correct for your system
weights_path = "/home/shaurya/Downloads/yolov8n_pokemon_more_scenes/kaggle/working/runs/detect/yolov8n_pokemon/weights/best.pt"
images_dir = "/home/shaurya/Pokemon_HACK/test_images/the-poke-war-hackathon-ai-guild-recuritment-hack/test_images"
json_file = "pokemon_kill_orders_spacy_v2.json"
output_csv = "pokemon_refined_centers_contour.csv"
annotated_dir = "annotated_refined_centers"
masks_dir = "annotated_masks" # New directory for masks

# --- SETUP ---
os.makedirs(annotated_dir, exist_ok=True)
os.makedirs(masks_dir, exist_ok=True)
model = YOLO(weights_path)

# Load JSON mapping (image_name -> target_pokemon)
try:
    with open(json_file, "r") as f:
        mappings = json.load(f)
except FileNotFoundError:
    print(f"Error: JSON file not found at {json_file}")
    exit()

rows = []
print("Starting Pokémon center refinement process...")

# --- PROCESSING LOOP ---
for image_name, target_pokemon in mappings.items():
    image_path = os.path.join(images_dir, image_name)
    if not os.path.exists(image_path):
        print(f"Warning: Skipping missing image -> {image_name}")
        continue

    try:
        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
    except Exception as e:
        print(f"Error opening image {image_name}: {e}")
        continue

    # --- First pass YOLO to find the Pokémon ---
    results = model(image_path, conf=0.4, verbose=False)

    centers = []

    for r in results:
        boxes = r.boxes
        names = r.names

        for box in boxes:
            cls_id = int(box.cls[0].item())
            cls_name = names[cls_id]

            if cls_name.lower() == target_pokemon.lower():
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                # Crop the region of interest from the original image
                cropped_pil = img.crop((x1, y1, x2, y2))
                
                # --- NEW: REFINE CENTER WITH GRABCUT SEGMENTATION ---
                refined_center = None
                try:
                    # Convert PIL crop to OpenCV format
                    cropped_cv = cv2.cvtColor(np.array(cropped_pil), cv2.COLOR_RGB2BGR)

                    # Create a mask for grabCut. Everything is initialized as probable background.
                    mask = np.zeros(cropped_cv.shape[:2], np.uint8)
                    
                    # These are temporary arrays used by the algorithm
                    bgdModel = np.zeros((1, 65), np.float64)
                    fgdModel = np.zeros((1, 65), np.float64)
                    
                    # The rectangle for grabCut is the entire cropped image
                    rect = (1, 1, cropped_cv.shape[1] - 2, cropped_cv.shape[0] - 2)
                    
                    # Run grabCut
                    cv2.grabCut(cropped_cv, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
                    
                    # Create a final binary mask where definite and probable foreground are 1, others are 0
                    binary_mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8') * 255

                    # --- SAVE THE NEW, CLEANER MASK ---
                    base_name, _ = os.path.splitext(image_name)
                    mask_filename = f"{base_name}_{target_pokemon}_{len(centers)}_mask.png"
                    mask_save_path = os.path.join(masks_dir, mask_filename)
                    cv2.imwrite(mask_save_path, binary_mask)
                    # --- END SAVE ---

                    # Find all contours in the new mask
                    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    if contours:
                        # Find the largest contour by area (should be the Pokémon)
                        largest_contour = max(contours, key=cv2.contourArea)
                        
                        # Calculate the moments of the largest contour
                        M = cv2.moments(largest_contour)
                        
                        # Calculate centroid (center of mass) if area is not zero
                        if M["m00"] != 0:
                            cx_local = int(M["m10"] / M["m00"])
                            cy_local = int(M["m01"] / M["m00"])
                            
                            # Convert local centroid coordinates to original image coordinates
                            cx_global = round(cx_local + x1, 2)
                            cy_global = round(cy_local + y1, 2)
                            refined_center = [cx_global, cy_global]

                except Exception as e:
                    print(f"Could not process contours for {image_name}. Error: {e}")

                # --- Fallback to bounding box center if contour method fails ---
                if not refined_center:
                    cx = round((x1 + x2) / 2, 2)
                    cy = round((y1 + y2) / 2, 2)
                    refined_center = [cx, cy]
                    
                centers.append(refined_center)

                # --- Annotate the refined center on the image ---
                cx, cy = refined_center
                radius = 5
                # Bounding box for the text
                text_bbox = draw.textbbox((cx + radius + 2, cy - radius), target_pokemon)
                draw.rectangle(text_bbox, fill="black")
                # Draw ellipse and text
                draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill="red", outline="red")
                draw.text((cx + radius + 2, cy - radius), target_pokemon, fill="white")


    rows.append([image_name, json.dumps(centers)])

    # Save annotated image
    save_path = os.path.join(annotated_dir, image_name)
    img.save(save_path)

# --- SAVE RESULTS ---
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image_id", "points"])
    writer.writerows(rows)

print(f"\n✅ Processing complete!")
print(f"✅ CSV saved to: {output_csv}")
print(f"✅ Annotated images saved in: {annotated_dir}/")
print(f"✅ Masks saved in: {masks_dir}/")

