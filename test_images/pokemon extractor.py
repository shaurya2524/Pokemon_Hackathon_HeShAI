import json

def extract_kill_orders(file_path):
    """
    Loads a JSON file and extracts the base Pokémon name from explicit termination orders.
    """
    # 1. DEFINE KEYWORDS & MAPPING
    # ----------------------------
    termination_words = [
        "terminate", "kill", "take down", "eliminate", "destroy",
        "neutralize", "remove", "wipe out"
    ]

    pokemon_mapping = {
        "pikachu": ["Pikachu", "electric rat", "yellow mouse", "tiny thunder beast", "rodent of sparks"],
        "charizard": ["Charizard", "winged inferno", "flame dragon", "scaled fire titan", "orange lizard"],
        "bulbasaur": ["Bulbasaur", "sprout toad", "green seedling", "plant reptile", "vine beast"],
        "mewtwo": ["Mewtwo", "synthetic mind weapon", "telekinetic predator", "psychic clone", "genetic experiment"]
    }

    results = {}

    # 2. LOAD AND PROCESS THE FILE
    # ----------------------------
    try:
        with open(file_path, 'r', encoding='latin-1') as f:
            json_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: The file '{file_path}' is not a valid JSON file.")
        return None

    # Loop through each record in the JSON data
    for record in json_data:
        image_id = record.get("image_id")
        prompt = record.get("prompt")

        if not image_id or not prompt:
            continue

        # 3. SCAN FOR ORDERS
        # ------------------
        sentences = prompt.replace(":", ".").replace(";", ".").split('.')
        
        found_order_for_record = False
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            has_termination_word = any(word in sentence_lower for word in termination_words)
            
            if has_termination_word:
                for pokemon_name, aliases in pokemon_mapping.items():
                    for alias in aliases:
                        if alias.lower() in sentence_lower:
                            results[image_id] = pokemon_name
                            found_order_for_record = True
                            break
                    if found_order_for_record:
                        break
            
            if found_order_for_record:
                break

        if not found_order_for_record:
            results[image_id] = "No explicit termination order found"

    return results

# --- HOW TO USE ---
if __name__ == "__main__":
    # Input file path
    input_json_file = '/home/shaurya/HACKATHON_2/test_images/the-poke-war-hackathon-ai-guild-recuritment-hack/test_prompts_orders.json'
    
    # --- CHANGED: Define an output filename ---
    output_json_file = 'pokemon_kill_orders.json'
    
    kill_orders = extract_kill_orders(input_json_file)
    
    if kill_orders:
        # --- CHANGED: Instead of printing, save the results to a file ---
        with open(output_json_file, 'w') as f:
            # json.dump writes the dictionary to the file
            # indent=4 makes the JSON file nicely formatted and easy to read
            json.dump(kill_orders, f, indent=4)
        
        print(f"✅ Success! The results have been saved to '{output_json_file}'")