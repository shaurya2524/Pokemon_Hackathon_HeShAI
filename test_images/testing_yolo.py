import os
from ultralytics import YOLO

# --- CONFIG ---
weights_path = "/home/shaurya/Downloads/yolov8n_pokemon_more_scenes/kaggle/working/runs/detect/yolov8n_pokemon/weights/best.pt"   # your pretrained YOLO weights
images_dir = "/home/shaurya/HACKATHON_2/test_images/the-poke-war-hackathon-ai-guild-recuritment-hack/test_images"      # folder containing images
save_dir = "./runs/detect/test_results"   # output folder for annotated images

# Load model
model = YOLO(weights_path)

# Run inference on all images in the directory
results = model.predict(
    source=images_dir,   # directory with images
    save=True,           # save annotated images
    conf=0.6,            # filter out predictions below 0.7 confidence
    project="runs/detect", 
    name="test_results", # folder name inside project
    exist_ok=True        # overwrite if folder exists
)

print(f"Annotated images are saved at: {os.path.abspath(save_dir)}")

