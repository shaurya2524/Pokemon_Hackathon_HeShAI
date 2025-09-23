import os
import json
import csv
from ultralytics import YOLO

# --- CONFIG ---
weights_path = "/home/shaurya/Downloads/yolov8n_pokemon_more_scenes/kaggle/working/runs/detect/yolov8n_pokemon/weights/best.pt"   # your pretrained YOLO weights
images_dir = "/home/shaurya/Pokemon_HACK/test_images/the-poke-war-hackathon-ai-guild-recuritment-hack/test_images"           # folder with images
json_file = "/home/shaurya/Downloads/longformer_prediction_30(1).json"      # mapping file
output_csv = "pokemon_trained_longformer_more_scenes_0.25.csv"       # final CSV file

model = YOLO(weights_path)

# Load JSON mapping (dict: image_name -> pokemon)
with open(json_file, "r") as f:
    mappings = json.load(f)

rows = []

for image_name, target_pokemon in mappings.items():
    image_path = os.path.join(images_dir, image_name)
    # breakpoint()
    if not os.path.exists(image_path):
        print(f"Skipping missing image: {image_name}")
        continue

    # Run YOLO inference
    results = model(image_path, conf=0.25, verbose=False)


    centers = []
    for r in results:
        boxes = r.boxes
        names = r.names  # class_id -> class_name mapping

        for box in boxes:
            cls_id = int(box.cls[0].item())
            cls_name = names[cls_id]

            if cls_name.lower() == target_pokemon.lower():
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx = round((x1 + x2) / 2, 2)
                cy = round((y1 + y2) / 2, 2)
                centers.append([cx, cy])

    rows.append([image_name, json.dumps(centers)])

# Write CSV
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image_id", "points"])
    writer.writerows(rows)

print(f"✅ CSV saved to {output_csv}")
