import os
import json
import csv
from ultralytics import YOLO
from PIL import Image, ImageDraw

# --- CONFIG ---
weights_path = "/home/shaurya/Downloads/yolov8n_pokemon_more_scenes/kaggle/working/runs/detect/yolov8n_pokemon/weights/best.pt"
images_dir = "/home/shaurya/Pokemon_HACK/test_images/the-poke-war-hackathon-ai-guild-recuritment-hack/test_images"
json_file = "pokemon_kill_orders_spacy_v2.json"      # mapping file
output_csv = "pokemon_trained_spacy_more_scenes_0.4_0.85_0.9_refined.csv"
annotated_dir = "refined_centers_thrice"  # folder where annotated images will be saved

os.makedirs(annotated_dir, exist_ok=True)

model = YOLO(weights_path)

# Load JSON mapping (dict: image_name -> pokemon)
with open(json_file, "r") as f:
    mappings = json.load(f)

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

    for r in results:
        boxes = r.boxes
        names = r.names

        for box in boxes:
            cls_id = int(box.cls[0].item())
            cls_name = names[cls_id]

            if cls_name.lower() == target_pokemon.lower():
                # Get first-pass box
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                # Crop region of interest
                cropped = img.crop((x1, y1, x2, y2))

                # --- Second pass YOLO on cropped image ---
                second_results = model(cropped, conf=0.85, verbose=False)

                refined_center = None
                for sr in second_results:
                    for sbox in sr.boxes:
                        s_cls_id = int(sbox.cls[0].item())
                        s_cls_name = sr.names[s_cls_id]

                        if s_cls_name.lower() == target_pokemon.lower():
                            sx1, sy1, sx2, sy2 = sbox.xyxy[0].tolist()

                            # Convert to original coords
                            sx1 += x1
                            sx2 += x1
                            sy1 += y1
                            sy2 += y1

                            # Crop again for third pass
                            second_crop = img.crop((int(sx1), int(sy1), int(sx2), int(sy2)))

                            # --- Third pass YOLO on cropped again ---
                            third_results = model(second_crop, conf=0.9, verbose=False)

                            for tr in third_results:
                                for tbox in tr.boxes:
                                    t_cls_id = int(tbox.cls[0].item())
                                    t_cls_name = tr.names[t_cls_id]

                                    if t_cls_name.lower() == target_pokemon.lower():
                                        tx1, ty1, tx2, ty2 = tbox.xyxy[0].tolist()

                                        # Convert back to original coords
                                        tx1 += sx1
                                        tx2 += sx1
                                        ty1 += sy1
                                        ty2 += sy1

                                        cx = round((tx1 + tx2) / 2, 2)
                                        cy = round((ty1 + ty2) / 2, 2)
                                        refined_center = [cx, cy]

                # If third pass didn't detect → fallback to second → else first
                if refined_center:
                    centers.append(refined_center)
                else:
                    cx = round((x1 + x2) / 2, 2)
                    cy = round((y1 + y2) / 2, 2)
                    centers.append([cx, cy])
                    refined_center = [cx, cy]

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
