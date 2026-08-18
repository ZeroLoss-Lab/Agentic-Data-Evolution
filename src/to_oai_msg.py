"""Convert question-answer JSONL records to OpenAI chat-message JSONL."""

import json
import os

# Current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

def process_single_file(input_file, output_file, system_prompt):
    """
    Convert one input file to the chat-message format.
    """
    count = 0
    try:
        with open(input_file, 'r', encoding='utf-8') as f_in, \
             open(output_file, 'w', encoding='utf-8') as f_out:
            
            for line_num, line in enumerate(f_in, 1):
                try:
                    line = line.strip()
                    if not line: continue
                    
                    item = json.loads(line)
                    
                    # Extract fields.
                    user_content = item.get('question', '')
                    assistant_content = item.get('answer', '')
                    
                    if not user_content or not assistant_content:
                        continue

                    # Build the target format.
                    new_entry = {
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_content},
                            {"role": "assistant", "content": assistant_content}
                        ]
                    }
                    
                    f_out.write(json.dumps(new_entry, ensure_ascii=False) + '\n')
                    count += 1
                    
                except json.JSONDecodeError:
                    print(f"  [WARNING] Skipping malformed JSON in {os.path.basename(input_file)} at line {line_num}.")
                except Exception as e:
                    print(f"  [ERROR] Failed to process line {line_num}: {e}")

        print(f"  -> Completed: {os.path.basename(output_file)} ({count} records processed)")

    except Exception as e:
        print(f"Unable to open {input_file}: {e}")

def process_folder(input_dir, output_dir, system_prompt):
    """
    Convert all JSON or JSONL files in a directory.
    """
    # Create the output directory when needed.
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        except OSError as e:
            print(f"Failed to create output directory: {e}")
            return

    # Collect all input files.
    if not os.path.exists(input_dir):
        print(f"[ERROR] Input directory does not exist: {input_dir}")
        return

    files = os.listdir(input_dir)
    # Keep JSON and JSONL files only.
    json_files = [f for f in files if f.endswith('.json') or f.endswith('.jsonl')]
    
    if not json_files:
        print(f"No .json or .jsonl files found in {input_dir}.")
        return

    print(f"Starting conversion for {len(json_files)} files...")
    print("-" * 50)

    # Convert each file.
    for filename in json_files:
        src_path = os.path.join(input_dir, filename)
        dst_path = os.path.join(output_dir, filename)
        
        print(f"Processing: {filename} ...")
        process_single_file(src_path, dst_path, system_prompt)
    
    print("-" * 50)
    print(f"All conversions completed. Results are in: {output_dir}")

# --- Configuration ---

# Required environment variables
input_folder_path = os.getenv("CONVERT_INPUT_DIR")
if not input_folder_path:
    raise ValueError("Error: CONVERT_INPUT_DIR environment variable not set")

output_folder_path = os.getenv("CONVERT_OUTPUT_DIR")
if not output_folder_path:
    raise ValueError("Error: CONVERT_OUTPUT_DIR environment variable not set")

# System prompt (with safe default)
default_system_prompt = os.getenv("CONVERT_SYSTEM_PROMPT", "You are a helpful assistant.")

# --- Entry point ---
if __name__ == "__main__":
    process_folder(input_folder_path, output_folder_path, default_system_prompt)
