import os
import random
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import argparse
from tqdm import tqdm

# --- Custom Pokémon mapping ---
pokemon_map = {
    "pikachu": 0,
    "charizard": 1,
    "bulbasaur": 2,
    "mewtwo": 3
}

import numpy as np

def apply_augmentations_and_shadow(pokemon_img, scenery_img, scenery_size, min_scale, max_scale_ratio):
    scenery_w, scenery_h = scenery_size
    pk_w, pk_h = pokemon_img.size

    # --- Scaling (same as before) ---
    rotation_expansion_factor = 1.5
    scale_w = (scenery_w * max_scale_ratio) / (pk_w * rotation_expansion_factor)
    scale_h = (scenery_h * max_scale_ratio) / (pk_h * rotation_expansion_factor)
    dynamic_max_scale = min(scale_w, scale_h)

    scale = dynamic_max_scale if dynamic_max_scale <= min_scale else random.uniform(min_scale, dynamic_max_scale)
    new_w, new_h = int(pk_w * scale), int(pk_h * scale)
    if new_w < 1 or new_h < 1:
        return None, None

    pokemon_resized = pokemon_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # --- Rotation & mirror ---
    angle = random.randint(-45, 45)
    pokemon_transformed = pokemon_resized.rotate(angle, expand=True, fillcolor=(0,0,0,0))
    if random.random() > 0.5:
        pokemon_transformed = ImageOps.mirror(pokemon_transformed)

    # --- Blending Augmentations ---
    # Match brightness/contrast to scenery
    scenery_np = np.array(scenery_img.convert("L"))
    scenery_brightness = np.mean(scenery_np) / 128.0  # normalize
    enhancer = ImageEnhance.Brightness(pokemon_transformed)
    pokemon_transformed = enhancer.enhance(random.uniform(0.8, 1.2) * scenery_brightness)

    enhancer = ImageEnhance.Contrast(pokemon_transformed)
    pokemon_transformed = enhancer.enhance(random.uniform(0.8, 1.2))

    # Slight Gaussian noise
    if random.random() > 0.5:
        arr = np.array(pokemon_transformed)
        noise = np.random.normal(0, 5, arr.shape).astype(np.int16)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        pokemon_transformed = Image.fromarray(arr)

    # Perspective warp (alignment variation)
    if random.random() > 0.5:
        width, height = pokemon_transformed.size
        coeffs = [
            1, random.uniform(-0.2, 0.2), 0,
            random.uniform(-0.2, 0.2), 1, 0
        ]
        pokemon_transformed = pokemon_transformed.transform(
            (width, height), Image.AFFINE, coeffs, resample=Image.Resampling.BICUBIC
        )
    # --- Ensure RGBA for alpha mask ---
    if pokemon_transformed.mode != "RGBA":
        pokemon_transformed = pokemon_transformed.convert("RGBA")
    # --- Shadow with randomized offset ---
    shadow_color = (0, 0, 0, 100)
    shadow_img = Image.new('RGBA', pokemon_transformed.size, shadow_color)
    alpha_mask = pokemon_transformed.getchannel('A')
    shadow_img.putalpha(alpha_mask)
    shadow_img = shadow_img.filter(ImageFilter.GaussianBlur(radius=5))

    shadow_offset_x = random.randint(5, 15)
    shadow_offset_y = random.randint(5, 15)

    return pokemon_transformed, shadow_img, (shadow_offset_x, shadow_offset_y)


def generate_dataset(pokemon_dir, scenery_dir, output_dir, 
                     num_variations_per_scenery=20, 
                     min_pokemon_per_image=4, 
                     max_pokemon_per_image=15, 
                     max_scale_ratio=0.6, 
                     min_scale=0.05,
                     apply_bokeh=False):
    
    output_images_dir = os.path.join(output_dir, "images")
    output_labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    # --- Load classes ---
    classes = [cls for cls in pokemon_map.keys()]
    
    # Prepare pools of all Pokémon crops for each class
    class_pokemon_files = {}
    for cls in classes:
        cls_path = os.path.join(pokemon_dir, cls)
        files = [os.path.join(cls_path, f) for f in os.listdir(cls_path)]
        if not files:
            print(f"Warning: No Pokémon images found in class '{cls}'")
        random.shuffle(files)
        class_pokemon_files[cls] = files.copy()

    scenery_files = [f for f in os.listdir(scenery_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg','.webp'))]
    if not scenery_files:
        print(f"Error: No scenery images found in '{scenery_dir}'.")
        return

    print(f"Found {len(classes)} classes: {', '.join(classes)}")
    print(f"Generating {num_variations_per_scenery} variations for each of the {len(scenery_files)} scenery images.")

    total_generated_count = 0

    for scenery_file in tqdm(scenery_files, desc="Processing Scenery"):
        scenery_path = os.path.join(scenery_dir, scenery_file)

        for variation in range(num_variations_per_scenery):
            base_scenery_img = Image.open(scenery_path).convert("RGBA")

            if apply_bokeh:
                bokeh_radius = random.uniform(2.0, 5.0)
                base_scenery_img = base_scenery_img.filter(ImageFilter.GaussianBlur(radius=bokeh_radius))

            scenery_w, scenery_h = base_scenery_img.size
            labels = []
            existing_bboxes = []

            avg_pokemon_per_image = (min_pokemon_per_image + max_pokemon_per_image) // 2
            num_to_place = max(min_pokemon_per_image, min(max_pokemon_per_image,
                                avg_pokemon_per_image + random.randint(-2, 2)))

            failed_placements = 0

            while len(existing_bboxes) < num_to_place and failed_placements < 100:
                cls = random.choice(classes)

                if not class_pokemon_files[cls]:
                    cls_path = os.path.join(pokemon_dir, cls)
                    class_pokemon_files[cls] = [os.path.join(cls_path, f) for f in os.listdir(cls_path)]
                    random.shuffle(class_pokemon_files[cls])

                pokemon_path = class_pokemon_files[cls].pop()
                pokemon_img = Image.open(pokemon_path).convert("RGBA")

                pokemon_final, shadow_final, shadow_offset = apply_augmentations_and_shadow(
                pokemon_img, base_scenery_img, (scenery_w, scenery_h), min_scale, max_scale_ratio
                )           


                if pokemon_final is None:
                    failed_placements += 1
                    continue

                pk_final_w, pk_final_h = pokemon_final.size
                max_x, max_y = scenery_w - pk_final_w, scenery_h - pk_final_h
                if max_x < 0 or max_y < 0:
                    failed_placements += 1
                    continue

                found_position = False
                for _ in range(50):
                    x, y = random.randint(0, max_x), random.randint(0, max_y)
                    new_bbox = (x, y, x + pk_final_w, y + pk_final_h)

                    is_overlapping = False
                    for existing_bbox in existing_bboxes:
                        ix1, iy1 = max(new_bbox[0], existing_bbox[0]), max(new_bbox[1], existing_bbox[1])
                        ix2, iy2 = min(new_bbox[2], existing_bbox[2]), min(new_bbox[3], existing_bbox[3])
                        inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                        if inter_area / (pk_final_w * pk_final_h) > 0.15:
                            is_overlapping = True
                            break

                    if not is_overlapping:
                        found_position = True
                        break

                if found_position:
                    shadow_offset_x, shadow_offset_y = 10, 10
                    shadow_pos = (x + shadow_offset_x, y + shadow_offset_y)
                    if shadow_pos[0] < scenery_w and shadow_pos[1] < scenery_h:
                        base_scenery_img.paste(shadow_final, shadow_pos, shadow_final)

                    base_scenery_img.paste(pokemon_final, (x, y), pokemon_final)

                    existing_bboxes.append(new_bbox)
                    x_center, y_center = (new_bbox[0] + new_bbox[2]) / 2 / scenery_w, (new_bbox[1] + new_bbox[3]) / 2 / scenery_h
                    width, height = (new_bbox[2] - new_bbox[0]) / scenery_w, (new_bbox[3] - new_bbox[1]) / scenery_h
                    class_idx = pokemon_map[cls]
                    labels.append(f"{class_idx} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                    failed_placements = 0
                else:
                    failed_placements += 1

            if len(existing_bboxes) >= min_pokemon_per_image:
                output_filename = f"{total_generated_count:04d}"
                out_img_path = os.path.join(output_images_dir, f"{output_filename}.jpg")
                base_scenery_img.convert("RGB").save(out_img_path, "JPEG")

                out_label_path = os.path.join(output_labels_dir, f"{output_filename}.txt")
                with open(out_label_path, "w") as f:
                    f.write("\n".join(labels))
                total_generated_count += 1
            else:
                tqdm.write(f"  > Skipped variation {variation + 1}/{num_variations_per_scenery} for {scenery_file}. Could only place {len(existing_bboxes)} Pokémon.")

    print(f"\nDataset generation complete! Generated {total_generated_count} total images.")
    print(f"Images are in: {output_images_dir}")
    print(f"Labels are in: {output_labels_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate a synthetic dataset for object detection.")
    parser.add_argument("--pokemon_dir", type=str, required=True, help="Path to Pokémon crop subfolders.")
    parser.add_argument("--scenery_dir", type=str, required=True, help="Path to scenery images.")
    parser.add_argument("--output_dir", type=str, default="generated_dataset", help="Directory to save the dataset.")
    parser.add_argument("--num_variations", type=int, default=20, help="Variations to generate per scenery image.")
    parser.add_argument("--min_pokemon", type=int, default=4, help="Minimum Pokémon per image.")
    parser.add_argument("--max_pokemon", type=int, default=15, help="Maximum Pokémon per image.")
    parser.add_argument("--max_scale_ratio", type=float, default=0.6, help="Max size of a Pokémon relative to the scenery's smaller dimension (e.g., 0.6 = 60%).")
    parser.add_argument("--min_scale", type=float, default=0.05, help="Min absolute scaling factor for a Pokémon.")
    parser.add_argument("--bokeh", action="store_true", help="Apply a random bokeh (background blur) effect to scenery images.")
    
    args = parser.parse_args()
    
    generate_dataset(
        pokemon_dir=args.pokemon_dir,
        scenery_dir=args.scenery_dir,
        output_dir=args.output_dir,
        num_variations_per_scenery=args.num_variations,
        min_pokemon_per_image=args.min_pokemon,
        max_pokemon_per_image=args.max_pokemon,
        max_scale_ratio=args.max_scale_ratio,
        min_scale=args.min_scale,
        apply_bokeh=args.bokeh
    )

