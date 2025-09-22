import os
import json
import asyncio
from pathlib import Path
from typing import List, Dict, Any

# pip install ultralytics groq pandas transformers torch
# Note: 'torch' or 'tensorflow' is required for transformers
import pandas as pd
from groq import AsyncGroq
from ultralytics import YOLO
from transformers import pipeline

# ---------- CONFIGURATION ----------
PROMPTS_JSON = "/home/shaurya/HACKATHON_2/yolo_approach_own_prompts/own_prompts.json"
IMAGE_ROOT = "/home/shaurya/HACKATHON_2/dataset/images"
YOLO_WEIGHTS = "/home/shaurya/HACKATHON_2/yolo_approach_own_prompts/runs/yolov8n_custom4/weights/weights_16_09/best.pt"
OUTPUT_CSV = "./detections_by_ownprompt_robust.csv"
GROQ_MODEL = "qwen/qwen3-32b"
MIN_SCORE = 0.3  # Minimum confidence score to accept a YOLO detection.
BERT_CONFIDENCE_THRESHOLD = 0.6 # Confidence needed for the local BERT model to accept a classification.
# ---------- INITIALIZE MODELS ----------
# GROQ API Client (remote)
client = AsyncGroq()  # Expects GROQ_API_KEY environment variable

# YOLOv8 Object Detector (local)
yolo_model = YOLO(YOLO_WEIGHTS)
print(f"✅ YOLO model loaded from: {YOLO_WEIGHTS}")

# BERT-based Zero-Shot Classifier (local)
# This will download the model from Hugging Face on the first run.
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
print("✅ BERT-based zero-shot classifier initialized (this may download the model on first run).")


# ---------- EXTRACTOR FUNCTIONS ----------

async def extract_pokemon_with_groq(prompt: str) -> str | None:
    """
    (API-BASED) Uses Groq to extract a Pokémon name if the prompt implies an "attack" command.
    """
    system_prompt = (
        "You are a strict JSON extractor. Check if the user's text is an explicit instruction "
        "to kill, terminate, eliminate, or attack a Pokémon. "
        "If yes, respond with {\"pokemon\": \"<name>\"} where <name> is one of: "
        "pikachu, charizard, bulbasaur, mewtwo. Otherwise, respond with {\"pokemon\": null}."
    )
    try:
        completion = await client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=500,
            response_format={"type": "json_object"},
        )
        response_text = completion.choices[0].message.content
        data = json.loads(response_text)
        pokemon = data.get("pokemon")
        return pokemon.strip().lower() if isinstance(pokemon, str) else None
    except Exception as e:
        print(f"  [WARN] Groq extraction or parsing failed: {e}")
        return None

def extract_pokemon_with_bert(prompt: str) -> str | None:
    """
    (LOCAL) Uses a zero-shot classification model to identify the target Pokémon.
    Returns the Pokémon name in lowercase or None if confidence is too low.
    """
    candidate_labels = ["pikachu", "charizard", "bulbasaur", "mewtwo"]
    hypothesis_template = "This text is about targeting to kill or execute or destroy in any manner {}." # Helps guide the model
    
    try:
        # The classifier runs synchronously, so we don't need 'await'
        results = classifier(prompt, candidate_labels, hypothesis_template=hypothesis_template, multi_label=False)
        
        top_score = results["scores"][0]
        if top_score >= BERT_CONFIDENCE_THRESHOLD:
            top_label = results["labels"][0]
            return top_label.lower()
        return None # Confidence was too low
    except Exception as e:
        print(f"  [WARN] BERT extraction failed: {e}")
        return None

# ---------- DETECTION FUNCTION ----------

def detect_objects_in_image(image_path: str) -> List[Dict[str, Any]]:
    """
    Runs YOLOv8 object detection on an image and returns a list of detected objects.
    """
    results = yolo_model.predict(source=image_path, save=False, verbose=False)
    if not results or not results[0].boxes:
        return []

    detections = []
    res = results[0]
    names = res.names

    for box in res.boxes:
        if box.conf[0] < MIN_SCORE:
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append({
            "class_name": names[int(box.cls[0])].lower(),
            "score": float(box.conf[0]),
            "center": [round((x1 + x2) / 2, 2), round((y1 + y2) / 2, 2)],
        })
    return detections

# ---------- MAIN ORCHESTRATION ----------

# ---------- MAIN ORCHESTRATION ----------

async def main():
    """Main orchestration function to process all prompts and images."""
    with open(PROMPTS_JSON, "r") as f:
        prompts_data = json.load(f)

    # List for the first CSV (with coordinates)
    output_rows = []
    
    # --- STEP 1: ADD THIS LINE ---
    # Create a new list for the second CSV (with the target Pokémon)
    target_pokemon_rows = []

    total_items = len(prompts_data)
    for i, item in enumerate(prompts_data):
        prompt = item.get("prompt", "")
        image_id = item.get("image_id") or item.get("image") or item.get("filename")

        print(f"[{i+1}/{total_items}] Processing: {image_id}")

        if not (prompt and image_id):
            print(f"  ⏭️  Skipping item due to missing prompt or image_id.")
            output_rows.append({"image_filename": image_id or "unknown", "coords": json.dumps([])})
            # Also add a blank row to our new list for completeness
            target_pokemon_rows.append({"image_filename": image_id or "unknown", "target_pokemon": None})
            continue

        image_path = Path(IMAGE_ROOT) / image_id
        coords = []

        # Using the local BERT-based zero-shot model
        target_pokemon = extract_pokemon_with_bert(prompt)
        
        # --- STEP 2: ADD THIS LINE ---
        # Store the result for our second CSV
        target_pokemon_rows.append({"image_filename": image_id, "target_pokemon": target_pokemon})
        
        print(f"  - Model target: {target_pokemon or 'None'}")

        if not image_path.exists():
            print(f"  [WARN] Image not found at: {image_path}. Storing empty coords.")
        elif target_pokemon:
            all_detections = detect_objects_in_image(str(image_path))
            coords = [
                d["center"] for d in all_detections
                if d["class_name"] == target_pokemon
            ]
        
        print(f"  - Found coordinates: {coords}")
        output_rows.append({"image_filename": image_id, "coords": json.dumps(coords)})

    # This saves your original CSV with the coordinates
    pd.DataFrame(output_rows).to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Successfully saved {len(output_rows)} detection results to {OUTPUT_CSV}")

    # --- STEP 3: ADD THESE LINES ---
    # Define a new filename for the target Pokémon CSV
    target_csv_filename = "./pokemon_targets_output.csv"
    # Save the new list to the new CSV file
    pd.DataFrame(target_pokemon_rows).to_csv(target_csv_filename, index=False)
    print(f"✅ Successfully saved {len(target_pokemon_rows)} target Pokémon records to {target_csv_filename}")

if __name__ == "__main__":
    asyncio.run(main())

