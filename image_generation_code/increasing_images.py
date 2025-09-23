import os
import cv2
import numpy as np
from PIL import Image, ImageDraw
import random
from tqdm import tqdm
import argparse

def add_bokeh(image_pil):
    """
    Adds a synthetic bokeh effect to a PIL image.

    Args:
        image_pil (PIL.Image.Image): The input image, which should be in 'RGBA' mode.

    Returns:
        PIL.Image.Image: The image with the bokeh effect, in 'RGB' mode.
    """
    # Create a transparent overlay of the same size as the image
    overlay = Image.new('RGBA', image_pil.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    width, height = image_pil.size

    # Configuration for the Bokeh Effect
    num_circles = random.randint(30, 60)
    min_radius = int(min(width, height) * 0.03)
    max_radius = int(min(width, height) * 0.12)

    for _ in range(num_circles):
        x = random.randint(0, width)
        y = random.randint(0, height)
        radius = random.randint(min_radius, max_radius)
        
        # Random warm color with random opacity
        r, g, b = random.randint(200, 255), random.randint(100, 220), random.randint(0, 50)
        alpha = random.randint(50, 150)
        color = (r, g, b, alpha)
        
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color)

    # Apply a heavy Gaussian blur to the overlay
    overlay_cv = cv2.cvtColor(np.array(overlay), cv2.COLOR_RGBA2BGRA)
    blur_amount = max_radius // 2 * 2 + 1
    blurred_overlay_cv = cv2.GaussianBlur(overlay_cv, (blur_amount, blur_amount), 0)
    blurred_overlay_pil = Image.fromarray(cv2.cvtColor(blurred_overlay_cv, cv2.COLOR_BGRA2RGBA))

    # Composite the blurred overlay onto the original image
    final_image = Image.alpha_composite(image_pil.convert('RGBA'), blurred_overlay_pil)

    return final_image.convert('RGB')


def process_images_and_labels(img_dir, label_dir, out_img_dir, out_label_dir, num_augmentations):
    """
    Processes images and labels to create multiple augmented versions of each image.
    For each version, it applies a two-step augmentation process:
    1. Augment a random object within its bounding box (flip, rotate, scale, translate).
    2. Apply a full-image bokeh effect.
    Labels are updated for augmentations that change bbox coordinates (scale, translate).
    """
    if not os.path.isdir(img_dir):
        print(f"\nError: Image directory not found at '{os.path.abspath(img_dir)}'")
        return
    if not os.path.isdir(label_dir):
        print(f"\nError: Label directory not found at '{os.path.abspath(label_dir)}'")
        return
        
    os.makedirs(out_img_dir, exist_ok=True)
    os.makedirs(out_label_dir, exist_ok=True)

    supported_formats = ('.png', '.jpg', '.jpeg')
    image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(supported_formats)]

    if not image_files:
        print(f"No images found in {img_dir}")
        return

    print(f"Found {len(image_files)} images. Generating {num_augmentations} augmented versions per image...")
    total_generated = 0

    # The main loop iterates through each original image file
    for filename in tqdm(image_files, desc="Processing Original Images"):
        base_name, _ = os.path.splitext(filename)
        img_path = os.path.join(img_dir, filename)
        label_path = os.path.join(label_dir, f"{base_name}.txt")
        
        if not os.path.exists(label_path):
            continue

        # --- NEW: Inner loop to create multiple augmentations per image ---
        for i in range(num_augmentations):
            try:
                # Always open the original image and labels for each new augmentation
                with Image.open(img_path) as img:
                    img = img.convert('RGBA')
                    img_w, img_h = img.size
                    
                    with open(label_path, 'r') as f:
                        lines = f.readlines()
                    if not lines: continue

                    # Step 1: Apply a random augmentation to a random object
                    line_idx_to_augment = random.randint(0, len(lines) - 1)
                    # Make a copy of the lines to modify for this iteration
                    current_lines = list(lines)
                    chosen_line = current_lines[line_idx_to_augment]
                    
                    parts = chosen_line.strip().split()
                    class_id, x_center_rel, y_center_rel, w_rel, h_rel = map(float, parts)
                    
                    w_abs, h_abs = w_rel * img_w, h_rel * img_h
                    x_center_abs, y_center_abs = x_center_rel * img_w, y_center_rel * img_h
                    x_min, y_min = int(x_center_abs - (w_abs / 2)), int(y_center_abs - (h_abs / 2))

                    augmentation_choice = random.choice(['flip', 'rotate', 'scale', 'translate'])
                    
                    cropped_object = img.crop((x_min, y_min, x_min + int(w_abs), y_min + int(h_abs)))
                    augmented_image = img.copy()

                    if augmentation_choice == 'flip':
                        augmented_object = cropped_object.transpose(Image.FLIP_LEFT_RIGHT)
                        augmented_image.paste(augmented_object, (x_min, y_min), augmented_object.convert('RGBA'))

                    elif augmentation_choice == 'rotate':
                        angle = random.uniform(-20, 20)
                        augmented_object = cropped_object.rotate(angle, expand=True, resample=Image.BICUBIC)
                        final_rotated = Image.new('RGBA', cropped_object.size, (0, 0, 0, 0))
                        paste_x = (cropped_object.width - augmented_object.width) // 2
                        paste_y = (cropped_object.height - augmented_object.height) // 2
                        final_rotated.paste(augmented_object, (paste_x, paste_y))
                        augmented_image.paste(final_rotated, (x_min, y_min), final_rotated.convert('RGBA'))

                    elif augmentation_choice == 'scale':
                        scale_factor = random.uniform(0.75, 1.25)
                        new_w, new_h = int(w_abs * scale_factor), int(h_abs * scale_factor)
                        scaled_obj = cropped_object.resize((new_w, new_h), Image.LANCZOS)
                        new_x_min = max(0, x_center_abs - new_w // 2)
                        new_y_min = max(0, y_center_abs - new_h // 2)
                        augmented_image.paste(scaled_obj, (int(new_x_min), int(new_y_min)), scaled_obj.convert('RGBA'))
                        new_x_center_rel, new_y_center_rel = (new_x_min + new_w / 2) / img_w, (new_y_min + new_h / 2) / img_h
                        new_w_rel, new_h_rel = new_w / img_w, new_h / img_h
                        current_lines[line_idx_to_augment] = f"{int(class_id)} {new_x_center_rel:.6f} {new_y_center_rel:.6f} {new_w_rel:.6f} {new_h_rel:.6f}\n"

                    elif augmentation_choice == 'translate':
                        dx, dy = int(random.uniform(-0.1, 0.1) * w_abs), int(random.uniform(-0.1, 0.1) * h_abs)
                        new_x_min = max(0, min(x_min + dx, img_w - w_abs))
                        new_y_min = max(0, min(y_min + dy, img_h - h_abs))
                        augmented_image.paste(cropped_object, (int(new_x_min), int(new_y_min)), cropped_object.convert('RGBA'))
                        new_x_center_rel, new_y_center_rel = (new_x_min + w_abs / 2) / img_w, (new_y_min + h_abs / 2) / img_h
                        current_lines[line_idx_to_augment] = f"{int(class_id)} {new_x_center_rel:.6f} {new_y_center_rel:.6f} {w_rel:.6f} {h_rel:.6f}\n"
                    
                    # Step 2: Apply Bokeh effect
                    final_augmented_image = add_bokeh(augmented_image)

                    # Step 3: Save results with unique names
                    out_img_path = os.path.join(out_img_dir, f"{base_name}_aug_{i}_{augmentation_choice}.jpg")
                    out_label_path = os.path.join(out_label_dir, f"{base_name}_aug_{i}_{augmentation_choice}.txt")
                    
                    final_augmented_image.save(out_img_path, 'JPEG')
                    
                    with open(out_label_path, 'w') as out_f:
                        out_f.writelines(current_lines)
                    total_generated += 1

            except Exception as e:
                print(f"Error processing {filename} on iteration {i}: {e}")
    
    print("\nProcessing complete!")
    print(f"Generated a total of {total_generated} new images and labels.")
    print(f"Augmented images saved in: {os.path.abspath(out_img_dir)}")
    print(f"Augmented labels saved in: {os.path.abspath(out_label_dir)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply combined augmentations to an object detection dataset.")
    parser.add_argument("--img_dir", type=str, required=True, help="Path to the directory with original images.")
    parser.add_argument("--label_dir", type=str, required=True, help="Path to the directory with YOLO format .txt label files.")
    parser.add_argument("--out_img_dir", type=str, required=True, help="Path to save augmented images.")
    parser.add_argument("--out_label_dir", type=str, required=True, help="Path to save corresponding label files.")
    # --- NEW ARGUMENT ---
    parser.add_argument("--num_augmentations", type=int, default=1, help="Number of augmented versions to create for each original image.")
    
    args = parser.parse_args()
    
    img_dir = os.path.expanduser(args.img_dir)
    label_dir = os.path.expanduser(args.label_dir)
    out_img_dir = os.path.expanduser(args.out_img_dir)
    out_label_dir = os.path.expanduser(args.out_label_dir)
    
    process_images_and_labels(img_dir, label_dir, out_img_dir, out_label_dir, args.num_augmentations)

