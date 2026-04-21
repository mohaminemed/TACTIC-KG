from ollama import Client
import json
import os
seen = "seen"  # "seen" or "unseen" depending on evaluation set
MODEL = "deepseek-v3.1:671b"
INPUT_DATASET = f"data/datasets/{seen}_test_dataset.json"
OUTPUT_DIR = f"outputs/{seen}/atomic_facts"
OUTPUT_DIR_CHUNKS = f"outputs/{seen}/semantic_chunks"
atomic = False

api_key = os.environ.get("OLLAMA_API_KEY")
if not api_key:
    raise ValueError("❌ Missing OLLAMA_API_KEY environment variable. Please set it before running.")

print("Ollama API Key detected.")

client = Client(
    host="https://ollama.com",
    headers={'Authorization': 'Bearer ' + api_key}
)

# ---------------------------------------------------------
#  OLLAMA Call
# ---------------------------------------------------------
def call_ollama(model: str, system_prompt: str, text_chunk: str):
    """Stream responses from Ollama for ATOM fact extraction."""
    messages = [
        { "role": "system", "content": system_prompt },
        { "role": "user",   "content": text_chunk },
    ]
    response = ""

    for part in client.chat(model, messages=messages, stream=True):
        delta = part.get("message", {}).get("content", "")
        response += delta
        print(delta, end="", flush=True)

    print()
    return response


# ---------------------------------------------------------
#  Prompt builder
# ---------------------------------------------------------
def build_atomic_prompt():
    return f"""
You are an ATOM-compliant extraction engine.

TASK:
Decompose the input CTI text into a list of **atomic**, **self-contained**, and **temporally-normalized** facts.

REQUIREMENTS:
- Each fact MUST express exactly one fact.
- Remove pronouns; use fully-qualified entity names.
- Convert ALL temporal references using the provided observation_date.
- No hallucinations or external knowledge.
- No redundancy.
- Output **JSON array only**, each item a string factoid.

OUTPUT FORMAT:
[
  "fact 1...",
  "fact 2...",
  ...
]
"""


# ---------------------------------------------------------
#  LLM Prompt Builder for Semantic Chunking
# ---------------------------------------------------------
def build_semantic_chunk_prompt(text: str):
    """
    Builds a prompt for an LLM to semantically chunk a CTI text.
    
    Args:
        text (str): The CTI text to chunk.
        observation_date (str): Date to normalize temporal references.
        max_tokens (int): Maximum tokens for the LLM (optional usage).
    
    Returns:
        str: LLM prompt string.
    """
    prompt = f"""
You are a semantic chunking engine for CTI texts.

TASK:
Decompose the input text into **self-contained** chunks.  
Each chunk should contain close-related facts.

REQUIREMENTS:
- Remove pronouns; use fully-qualified entity names.
- Preserve all details relevant to CTI analysis.
- Avoid hallucinations or assumptions beyond the text.
- Avoid redundancy.
- Ensure each chunk is readable as a standalone statement.

INPUT TEXT:
{text}

OUTPUT FORMAT:
[
  "chunk 1...",
  "chunk 2...",
  ...
]
"""
    return prompt

# ---------------------------------------------------------
#  Optional: semantic chunking 
# ---------------------------------------------------------
def semantic_chunk(text: str):
    """
    Semantic chunking logic.
    LLM-based chunker here.
    """

    prompt = build_semantic_chunk_prompt(text)

    response = call_ollama(
        model=MODEL,
        system_prompt=prompt,
        text_chunk=text
    )

    try:
            chunks = json.loads(response)

    except:
            print("⚠️ Failed to parse JSON. Saving raw output.")
      

    return chunks if isinstance(chunks, list) else [text]  # Fallback to original text if parsing fails


# ---------------------------------------------------------
#  MAIN EXECUTION
# ---------------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR_CHUNKS, exist_ok=True)

print(f"\n=== Loading dataset: {INPUT_DATASET} ===")

try:
    with open(INPUT_DATASET, "r", encoding="utf-8") as f:
        dataset = json.load(f)
        print(f"✅ Loaded dataset with {len(dataset)} entries.")
except Exception as e:
    raise RuntimeError(f"❌ Failed to load dataset file: {e}")

if not isinstance(dataset, list):
    raise ValueError("❌ Dataset must be a list of objects.")


prompt = build_atomic_prompt()

for item in dataset:
    source_id = item.get("source_id")
    report_text = item.get("text", "")

    if not source_id:
        print("⚠️ Missing source_id, skipping entry.")
        continue

    if not report_text:
        print(f"⚠️ Empty text for {source_id}, skipping.")
        continue

    output_path = os.path.join(OUTPUT_DIR, f"{source_id}.json") if atomic else os.path.join(OUTPUT_DIR_CHUNKS, f"{source_id}_chunks.json")

    print(f"\n=== Processing {source_id} ===")

    # ---------------------------------------------------------
    #  Semantic chunking 
    # ---------------------------------------------------------
    chunks = semantic_chunk(report_text)

    if atomic :
       all_atoms = []
       for idx, chunk in enumerate(chunks):
        print(f"\n--- Chunk {idx+1}/{len(chunks)} for {source_id} ---")
        response = call_ollama(
            model=MODEL,
            system_prompt=prompt,
            text_chunk=chunk
        )
        try:
            atoms = json.loads(response)
            if isinstance(atoms, list):
                all_atoms.extend(atoms)
            else:
                print("⚠️ Model did not return a list. Storing raw output.")
                all_atoms.append({"raw_output": response})
        except:
            print("⚠️ Failed to parse JSON. Saving raw output.")
            all_atoms.append({"raw_output": response})
        
        # ---------------------------------------------------------
        # Save Output
        # ---------------------------------------------------------
        with open(output_path, "w", encoding="utf-8") as out:
           json.dump(all_atoms, out, ensure_ascii=False, indent=2)

        print(f"✔️ Saved {len(all_atoms)} atomic facts to {output_path}")

    else:

        # ---------------------------------------------------------
        # Save Output
        # ---------------------------------------------------------
        with open(output_path, "w", encoding="utf-8") as out:
           json.dump(chunks, out, ensure_ascii=False, indent=2)

        print(f"✔️ Saved {len(chunks)} chunks to {output_path}")


print("\n🎉 Completed Chunking extraction for all reports.")
