import json
import math
from typing import Dict, Any, List
from PyPDF2 import PdfReader
from concurrent.futures import ThreadPoolExecutor

# --- Optional imports (only load what's needed) ---
# from openai import OpenAI
from langchain_community.llms import Ollama


# =========================
# CONFIGURATION
# =========================
CONFIG = {
    "llm_source": "local",   # "local" or "openai"
    "local_model": "gemma:2b",  # For Ollama
    "openai_model": "gpt-4o-mini",  # For cloud
    "max_chunk_size": 8000,  # characters per chunk
    "num_threads": 3          # Parallel chunk calls
}


# =========================
# PDF TEXT EXTRACTION
# =========================
def extract_text_from_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text += page_text + "\n"
    return text.strip()


# =========================
# CHUNKING
# =========================
def chunk_text(text: str, max_len: int) -> List[str]:
    """Split text into overlapping chunks."""
    chunks = []
    step = max_len - 1000  # small overlap
    for i in range(0, len(text), step):
        chunk = text[i:i + max_len]
        chunks.append(chunk)
    return chunks


# =========================
# PROMPT CREATION
# =========================
def build_prompt(contract_text: str, json_schema: Dict[str, Any], chunk_id: int) -> str:
    return f"""
You are an AI contract extraction assistant.

Extract fields as per this schema:
{json.dumps(json_schema, indent=2)}

This is part #{chunk_id} of a larger contract.
If information is not in this part, leave fields blank.

Contract Text (Chunk {chunk_id}):
-----------------
{contract_text}
-----------------

Return only a valid JSON object.
"""


# =========================
# MODEL CALLS
# =========================
def call_local_llm(prompt: str, model_name: str) -> str:
    llm = Ollama(model=model_name)
    return llm.invoke(prompt)


def call_openai_llm(prompt: str, model_name: str) -> str:
    from openai import OpenAI
    client = OpenAI()
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content.strip()


# =========================
# PARSE JSON SAFELY
# =========================
def safe_parse_json(raw_output: str) -> Dict[str, Any]:
    try:
        start = raw_output.find('{')
        end = raw_output.rfind('}') + 1
        return json.loads(raw_output[start:end])
    except Exception:
        return {}


# =========================
# CHUNK PROCESSING
# =========================
def process_chunk(chunk_data):
    chunk_text, json_schema, chunk_id = chunk_data
    prompt = build_prompt(chunk_text, json_schema, chunk_id)

    if CONFIG["llm_source"] == "local":
        raw_output = call_local_llm(prompt, CONFIG["local_model"])
    else:
        raw_output = call_openai_llm(prompt, CONFIG["openai_model"])

    return safe_parse_json(raw_output)


# =========================
# MERGE RESULTS
# =========================
def merge_chunk_results(results: List[Dict[str, Any]], schema: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple partial JSONs into a single result."""
    final = {k: "" for k in schema.keys()}

    for result in results:
        for key, value in result.items():
            if not final[key] and value:  # fill first non-empty
                final[key] = value

    return final


# =========================
# MAIN PIPELINE
# =========================
def extract_json_from_contract(pdf_path: str, json_schema: Dict[str, Any]) -> Dict[str, Any]:
    """Main pipeline to extract JSON from a potentially large contract."""
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text, CONFIG["max_chunk_size"])

    print(f"🔹 Total chunks created: {len(chunks)}")

    chunk_inputs = [(chunk, json_schema, i + 1) for i, chunk in enumerate(chunks)]

    # Process in parallel for efficiency
    with ThreadPoolExecutor(max_workers=CONFIG["num_threads"]) as executor:
        results = list(executor.map(process_chunk, chunk_inputs))

    final_json = merge_chunk_results(results, json_schema)
    return final_json


# =========================
# EXAMPLE USAGE
# =========================
if __name__ == "__main__":
    schema = {
        "party_1_name": "",
        "party_2_name": "",
        "effective_date": "",
        "termination_clause": "",
        "payment_terms": "",
        "governing_law": ""
    }

    pdf_path = "sample_contract.pdf"

    final_output = extract_json_from_contract(pdf_path, schema)
    print("\n✅ Final Extracted JSON:")
    print(json.dumps(final_output, indent=2))