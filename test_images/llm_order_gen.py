import json
import spacy
from spacy.matcher import PhraseMatcher
import sys
import time

# --- Configuration ---
INPUT_FILE_PATH = '/home/shaurya/HACKATHON_2/test_images/the-poke-war-hackathon-ai-guild-recuritment-hack/test_prompts_orders.json'
OUTPUT_FILE_PATH = 'pokemon_kill_orders_spacy_v2.json'

def extract_orders_with_spacy(file_path: str) -> dict:
    """
    Uses spaCy's PhraseMatcher and linguistic analysis to accurately extract termination orders.
    This version correctly handles multi-word aliases.
    """
    print("Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm")

    pokemon_mapping = {
        "pikachu": ["Pikachu", "electric rat", "yellow mouse", "tiny thunder beast", "rodent of sparks"],
        "charizard": ["Charizard", "winged inferno", "flame dragon", "scaled fire titan", "orange lizard"],
        "bulbasaur": ["Bulbasaur", "sprout toad", "green seedling", "plant reptile", "vine beast"],
        "mewtwo": ["Mewtwo", "synthetic mind weapon", "telekinetic predator", "psychic clone", "genetic experiment"]
    }

    # --- SETUP FOR PHRASE MATCHING ---
    # We use PhraseMatcher to correctly find multi-word aliases like "electric rat".
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = {}
    for base_name, aliases in pokemon_mapping.items():
        patterns[base_name] = [nlp.make_doc(alias) for alias in aliases]

    for base_name, docs in patterns.items():
        matcher.add(base_name, docs)

    # Base forms of termination verbs for checking context.
    termination_lemmas = {
        "terminate", "kill", "eliminate", "destroy", "neutralize", "remove", "wipe", "take", "engage"
    }
    
    # Load data
    try:
        with open(file_path, 'r', encoding='latin-1') as f:
            json_data = json.load(f)
    except Exception as e:
        print(f"❌ Error reading input file '{file_path}': {e}", file=sys.stderr)
        return {}

    final_results = {}
    total_records = len(json_data)
    start_time = time.time()
    
    prompts = [(record.get("prompt", ""), record.get("image_id")) for record in json_data]

    # Process all prompts efficiently using nlp.pipe
    for i, doc in enumerate(nlp.pipe([p[0] for p in prompts])):
        image_id = prompts[i][1]
        
        progress = f"Processing {image_id} ({i+1}/{total_records})"
        sys.stdout.write(f"\r{progress}")
        sys.stdout.flush()
        
        # 1. Find all Pokémon aliases in the text
        matches = matcher(doc)
        
        found_order = False
        for match_id, start, end in matches:
            pokemon_base_name = nlp.vocab.strings[match_id]
            pokemon_span = doc[start:end]

            # 2. Analyze the context BEFORE the found alias
            # Look at the 5 tokens preceding the match for a termination verb
            context_start = max(0, start - 5)
            context = doc[context_start:start]
            
            for token in reversed(context):
                if token.lemma_ in termination_lemmas:
                    final_results[image_id] = pokemon_base_name
                    found_order = True
                    break
            if found_order:
                break
        
        if not found_order:
            final_results[image_id] = "No explicit termination order found"
            
    end_time = time.time()
    print(f"\n\nProcessed {total_records} records in {end_time - start_time:.2f} seconds.")
    return final_results

# --- Main Execution Block ---
if __name__ == "__main__":
    print(f"Starting NLP extraction with spaCy v2 from '{INPUT_FILE_PATH}'...")
    
    kill_orders = extract_orders_with_spacy(INPUT_FILE_PATH)
    
    if kill_orders:
        try:
            with open(OUTPUT_FILE_PATH, 'w') as f:
                json.dump(kill_orders, f, indent=4)
            print(f"✅ Success! The results have been saved to '{OUTPUT_FILE_PATH}'")
        except Exception as e:
            print(f"❌ Error saving output file: {e}", file=sys.stderr)

