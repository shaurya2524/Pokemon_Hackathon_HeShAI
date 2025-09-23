import os
import cv2
import numpy as np
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor

# ------------------------------
# Paths
# ------------------------------
input_dir = "/home/shaurya/Pokemon_HACK/dataset/images"              # folder with input images
output_dir = "segmentation_maps/"  # folder to save results
os.makedirs(output_dir, exist_ok=True)

# ------------------------------
# Load YOLO model
# ------------------------------
yolo_model = YOLO("/home/shaurya/Downloads/yolov8n_pokemon_more_scenes/kaggle/working/runs/detect/yolov8n_pokemon/weights/best.pt")  # change to your model

# ------------------------------
# Load SAM
# ------------------------------
sam_checkpoint = "/home/shaurya/Pokemon_HACK/YOLO_WITH_SAM/sam_vit_b_01ec64.pth"  # download from Meta
model_type = "vit_b"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to("cuda")
predictor = SamPredictor(sam)

# ------------------------------
# Color function
# ------------------------------
def random_color(seed):
    np.random.seed(seed)
    return tuple(np.random.randint(0, 255, 3).tolist())

# ------------------------------
# Detect + segment + save
# ------------------------------
def process_image(image_path, save_path, conf=0.25):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = yolo_model.predict(image_rgb, conf=conf, device="cuda", verbose=False)[0]
    predictor.set_image(image_rgb)

    mask_overlay = np.zeros_like(image)  # pure segmentation map

    for i, box in enumerate(results.boxes):
        cls_id = int(box.cls.cpu().item())
        xyxy = box.xyxy.cpu().numpy().astype(int)[0]

        # SAM segmentation
        masks, _, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=xyxy,
            multimask_output=False,
        )

        mask = masks[0]
        color = random_color(cls_id)

        # Apply colored mask
        mask_overlay[mask] = color

    cv2.imwrite(save_path, mask_overlay)

# ------------------------------
# Loop through directory
# ------------------------------
for file in os.listdir(input_dir):
    if file.lower().endswith((".jpg", ".jpeg", ".png")):
        in_path = os.path.join(input_dir, file)
        out_path = os.path.join(output_dir, file)
        process_image(in_path, out_path)
        print(f"Saved segmentation map: {out_path}")
