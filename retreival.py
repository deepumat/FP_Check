from typing import List, Tuple
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# from anthropic import Anthropic  # if you want to really call Claude

TOP_K_CHUNKS = 5


def load_chunks(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["chunk_text"] = df["chunk_text"].astype(str)
    return df


def get_top_k_chunks(
    df: pd.DataFrame,
    question: str,
    k: int = TOP_K_CHUNKS
) -> List[Tuple[int, float]]:
    texts = df["chunk_text"].tolist()

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform([question] + texts)

    question_vec = tfidf_matrix[0:1]
    chunks_vec = tfidf_matrix[1:]

    sims = cosine_similarity(question_vec, chunks_vec)[0]

    df_scores = pd.DataFrame({
        "chunk_id": df["chunk_id"],
        "score": sims
    }).sort_values("score", ascending=False).head(k)

    return list(df_scores.itertuples(index=False, name=None))
    # [(chunk_id, score), ...]


def build_prompt(question: str, df: pd.DataFrame, top_chunks: List[Tuple[int, float]]) -> str:
    lines = []
    lines.append("You are a helpful assistant answering questions based ONLY on the provided document excerpts.")
    lines.append("If the answer is not contained in the excerpts, say you don't know.")
    lines.append("")
    lines.append(f"Question: {question}")
    lines.append("")
    lines.append("Relevant excerpts:")

    for rank, (chunk_id, score) in enumerate(top_chunks, start=1):
        row = df[df["chunk_id"] == chunk_id].iloc[0]
        lines.append(
            f"\n[Excerpt {rank} | chunk_id={chunk_id} | doc={row['doc_name']} "
            f"| pages {row['page_start']}-{row['page_end']} | score={score:.3f}]\n"
            f"{row['chunk_text']}"
        )

    lines.append("\nNow answer the question using ONLY the information in the excerpts above.")
    return "\n".join(lines)


def ask_claude(prompt: str) -> str:
    """
    Replace this stub with your actual Claude 3.5 call.
    """

    # Example with anthropic client (pseudo-code):
    # client = Anthropic(api_key="YOUR_KEY")
    # resp = client.messages.create(
    #     model="claude-3-5-sonnet-20241022",
    #     max_tokens=512,
    #     temperature=0.1,
    #     system="You are a careful assistant. Answer strictly based on the provided context.",
    #     messages=[{"role": "user", "content": prompt}],
    # )
    # return resp.content[0].text

    return f"(DEBUG) Would send this prompt to Claude:\n\n{prompt[:1500]}..."


def answer_question(csv_path: str, question: str) -> str:
    df = load_chunks(csv_path)
    top_chunks = get_top_k_chunks(df, question, k=TOP_K_CHUNKS)
    prompt = build_prompt(question, df, top_chunks)
    answer = ask_claude(prompt)
    return answer


if __name__ == "__main__":
    csv_path = "sample_chunks.csv"
    q = "What are the eligibility criteria mentioned in this document?"
    print(answer_question(csv_path, q))