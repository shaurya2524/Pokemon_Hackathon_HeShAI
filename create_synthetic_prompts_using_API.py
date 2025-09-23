import os
import random
import json
import time
from groq import Groq

try:
    # Best practice: Set your API key as an environment variable
    # In your terminal: export GROQ_API_KEY="YOUR_GROQ_API_KEY"
    client = Groq(api_key="API_KEY")
    print("Groq client initialized from environment variable.")
except Exception:
    print("Failed to initialize Groq client. Ensure GROQ_API_KEY is set.")
    # You can also hardcode it, but it's not recommended:
    # client = Groq(api_key="gsk_YourGroqApiKeyHere")
    exit()


# --- Your High-Quality Examples ---
# --- Configuration ---
json_file_path = r'/home/hemanthm/Desktop/HackVertical/Pokemon Hackathon/test_dataset/test_prompts_orders.json' # Make sure this path is correct
num_samples = 5 # The 'n' number of random examples you want

def get_good_prompt_examples():
    try:
        # Open and load the JSON file
        with open(json_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            all_prompts_data = json.load(f)

        # Check if the loaded data is a list and not empty
        if isinstance(all_prompts_data, list) and all_prompts_data:
            # Ensure we don't try to sample more items than exist in the list
            if len(all_prompts_data) < num_samples:
                print(f"Warning: Requested {num_samples} samples, but file only contains {len(all_prompts_data)}. Using all available prompts.")
                selected_samples = all_prompts_data
            else:
                # Select 'n' random dictionary objects from the list
                selected_samples = random.sample(all_prompts_data, num_samples)

            # Extract the 'prompt' string from each selected dictionary
            GOOD_PROMPT_EXAMPLES = [sample['prompt'] for sample in selected_samples]

        else:
            print("Error: The JSON file does not contain a list of prompts or is empty.")

    except FileNotFoundError:
        print(f"Error: The file '{json_file_path}' was not found.")
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Error reading or parsing the JSON file: {e}")
        print("Please ensure the file is a valid JSON and contains a list of objects, each with a 'prompt' key.")

    return GOOD_PROMPT_EXAMPLES

def generate_new_prompts(model, examples, target_pokemon):
    """
    Generates a new prompt using the Gemini model with few-shot examples.
    """
    # Create the instruction prompt for the model
    # This "meta-prompt" tells the model its job: to learn from examples and create a new one.
    prompt_for_model = f"""
    You are a creative AI assistant for generating military-style tactical briefings.
    Your task is to generate a new, unique tactical prompt in the same style as the examples provided.

    Here are some examples of high-quality prompts:
    ---
    EXAMPLE 1: {examples[0]}
    ---
    EXAMPLE 2: {examples[1]}
    ---
    EXAMPLE 3: {examples[2]}
    ---
    EXAMPLE 4: {examples[3]}
    ---
    EXAMPLE 5: {examples[4]}
    ---

    Now, based on the style, tone, and structure of the examples, create a completely new prompt with the following entities:

    - Target to Neutralize: {target_pokemon}

    [
        "terminate": "kill",
        "destroy": "kill",
        "eliminate": "kill",
        "take down": "kill",
        "neutralize": "kill",
        "neutralise": "kill",
        "remove": "kill",
        "wipe out": "kill",

        "electric rat": "pikachu",
        "yellow mouse": "pikachu",
        "rodent of sparks": "pikachu",
        "tiny thunder beast": "pikachu",

        "flame dragon": "charizard",
        "winged inferno": "charizard",
        "orange lizard": "charizard",
        "scaled fire titan": "charizard",

        "sprout toad": "bulbasaur",
        "vine beast": "bulbasaur",
        "green seedling": "bulbasaur",
        "plant reptile": "bulbasaur",

        "telekinetic predator": "mewtwo",
        "psychic clone": "mewtwo",
        "genetic experiment": "mewtwo",
        "synthetic mind weapon": "mewtwo",
    ]
    These are some of the synonyms that you can use to make it creative and varied.
    - Ensure the prompt is detailed, uses military jargon, and includes at least one explicit order.
    ALSO, ONLY THE SPECIFIED POKEMON SHOULD BE TARGETED. YOU CAN USE OTHER POKEMONS BUT DONT TARGET THEM. DONT DIRECTLY MENTION THE TARGET POKEMON, LIKE BLEND THAT INFORMATION IN THE PROMPT.
    DONT INCLUDE YOUR THINKING PROCESS. JUST GIVE THE PROMPT. ALSO, DONT MAKE IT TOO LONG. IDEALLY IT SHOULD BE ABOUT THE SAME LENGTH AS THE EXAMPLES.
    """
    system_prompt = "You are a creative AI assistant for generating military-style tactical briefings."
    try:
        # Groq uses a chat completions format
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": prompt_for_model,
                }
            ],
            model=model_name,
        )

        content = chat_completion.choices[0].message.content

        # 2. Check if content is not None before stripping
        if content:
            return content.strip()
        else:
            # If content is None, return an empty string or None to handle it later
            print("Warning: API returned an empty message.")
            return None
    except Exception as e:
        print(f"An error occurred during API call: {e}")
        return None

# --- Main Execution ---
if __name__ == "__main__":
    # MODIFICATION: Set the Groq model name
    model_name = "qwen/qwen3-32b"

    POKEMON_LIST = ["Pikachu", "Charizard", "Bulbasaur", "Mewtwo"]
    num_to_generate = 2
    
    generated_prompts = []

    print(f"Starting generation of {num_to_generate} new prompts using Groq and {model_name}...")
    for i in range(num_to_generate):
        target = random.choice(POKEMON_LIST)
        print(f"  > Generating prompt #{i+1} (Target: {target})")
        
        GOOD_PROMPT_EXAMPLES = get_good_prompt_examples()
        if not GOOD_PROMPT_EXAMPLES:
            print("    ...Failed to load examples. Skipping.")
            continue

        # Pass the Groq client and model_name to the function
        new_prompt = generate_new_prompts(client, model_name, target)
        
        if new_prompt:
            generated_prompts.append({
                "target": target,
                "prompt": new_prompt
            })
            print("    ...Success!")
        else:
            print("    ...Failed to generate.")
            
        time.sleep(1) # Rate limiting

    output_filename = f"newly_generated_prompts_{model_name.replace('/', '_')}_groq.json"
    with open(output_filename, 'w') as f:
        json.dump(generated_prompts, f, indent=2)

    print(f"\n✅ Generation complete! Saved {len(generated_prompts)} new prompts to '{output_filename}'.")