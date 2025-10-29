import fitz  # PyMuPDF
import json
from difflib import unified_diff
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer, util
import textwrap
import subprocess

# =======================
# PDF TEXT EXTRACTION
# =======================
def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text.strip()


# =======================
# CHUNKING FUNCTION
# =======================
def chunk_text(text, max_chars=4000):
    """Split long text into manageable chunks for Gemma."""
    paragraphs = text.split("\n")
    chunks, current_chunk = [], ""
    
    for p in paragraphs:
        if len(current_chunk) + len(p) > max_chars:
            chunks.append(current_chunk.strip())
            current_chunk = p
        else:
            current_chunk += "\n" + p
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks


# =======================
# GEMMA SUMMARIZATION (LOCAL)
# =======================
def call_local_gemma(prompt, model="gemma:2b"):
    """
    Modify this to your Gemma setup.
    Example below works with Ollama CLI.
    """
    try:
        cmd = ["ollama", "run", model, prompt]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        return result.stdout.strip()
    except Exception as e:
        return f"Error calling Gemma: {e}"


def summarize_with_gemma_long(text, model="gemma:2b"):
    """
    Hierarchical summarization:
    1. Summarize each chunk
    2. Summarize all partial summaries
    """
    chunks = chunk_text(text)
    partial_summaries = []

    for i, chunk in enumerate(chunks, 1):
        prompt = f"""
        You are a legal contract summarizer.
        Summarize this contract section {i}/{len(chunks)} focusing on:
        - Parties
        - Effective Date
        - Obligations
        - Termination
        - Payment Terms
        - Jurisdiction
        Provide both:
        1. JSON structure
        2. 100-word natural summary.
        
        Section Text:
        {chunk}
        """
        summary = call_local_gemma(prompt, model)
        partial_summaries.append(summary)

    combined_prompt = f"""
    Merge and refine these {len(partial_summaries)} section summaries into a single,
    comprehensive summary of the contract.
    Provide:
    1. Final structured JSON summary
    2. Final natural language summary under 200 words.

    Section Summaries:
    {'\n'.join(partial_summaries)}
    """
    final_summary = call_local_gemma(combined_prompt, model)
    return final_summary


# =======================
# TEXTUAL COMPARISON
# =======================
def compare_textual(contract_text1, contract_text2):
    diff = unified_diff(
        contract_text1.splitlines(),
        contract_text2.splitlines(),
        lineterm='',
        fromfile='Contract 1',
        tofile='Contract 2'
    )
    diff_text = "\n".join(diff)
    return diff_text if diff_text.strip() else "No textual differences found."


# =======================
# SEMANTIC COMPARISON
# =======================
def compare_semantic(contract_text1, contract_text2, threshold=0.95):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    emb1 = model.encode(contract_text1, convert_to_tensor=True)
    emb2 = model.encode(contract_text2, convert_to_tensor=True)
    similarity = util.cos_sim(emb1, emb2).item()

    result = {
        "semantic_similarity": round(similarity, 3),
        "are_meaningfully_same": similarity >= threshold
    }
    return result


# =======================
# MAIN FUNCTION
# =======================
def analyze_contracts(pdf1, pdf2=None):
    text1 = extract_text_from_pdf(pdf1)
    summary1 = summarize_with_gemma_long(text1)

    result = {
        "contract_1_summary": summary1,
    }

    if pdf2:
        text2 = extract_text_from_pdf(pdf2)
        summary2 = summarize_with_gemma_long(text2)
        text_diff = compare_textual(text1, text2)
        semantic_diff = compare_semantic(text1, text2)

        result.update({
            "contract_2_summary": summary2,
            "text_diff": text_diff,
            "semantic_diff": semantic_diff
        })

    return result


# =======================
# EXAMPLE USAGE
# =======================
if __name__ == "__main__":
    pdf1 = "contract_v1.pdf"
    pdf2 = "contract_v2.pdf"
    analysis = analyze_contracts(pdf1, pdf2)
    print(json.dumps(analysis, indent=2))