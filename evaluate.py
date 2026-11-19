# evaluate.py: measures how often the search finds the right page, for different chunk sizes
import os                                                   # for building file paths
import json                                                 # for reading the test questions
import chromadb                                             # the database that holds the chunks
from pypdf import PdfReader                                 # reads the PDFs
from sentence_transformers import SentenceTransformer       # turns text into numbers

BASE = os.path.dirname(os.path.abspath(__file__))           # the CityLens folder
DATA_DIR = os.path.join(BASE, "data")                       # where your PDFs live
TOP_K = 5                                                   # how many results count as "found"
CHUNK_SIZES = [100, 200, 400]                               # chunk sizes (in words) to compare

# Same number-making model as ingest.py and ask.py
model = SentenceTransformer("all-MiniLM-L6-v2")

# Load your test questions: each has a question, the right file, and the right page
with open(os.path.join(BASE, "eval", "test_questions.json")) as f:
    questions = json.load(f)

# Read every page of every PDF once, so we can re-chunk it at different sizes
pages = []   # list of (filename, page_number, text)
for filename in sorted(os.listdir(DATA_DIR)):
    if not filename.endswith(".pdf"):
        continue
    reader = PdfReader(os.path.join(DATA_DIR, filename))
    for page_num, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text()
        except Exception:
            continue    # skip pages that can't be read, same as ingest.py
        if text:
            pages.append((filename, page_num, text))


def chunk_text(text, size, overlap):
    """Cut one page into overlapping chunks of `size` words."""
    words = text.split()
    step = size - overlap                                    # how far to move for the next chunk
    for start in range(0, len(words), step):
        piece = " ".join(words[start:start + size])
        if len(piece.split()) > 30:                          # skip tiny scraps
            yield piece


def score(size):
    """Build a temporary index with this chunk size, then test every question.
    Returns (number of chunks, top-1 score, top-5 score)."""
    overlap = size // 5                                      # overlap is 20% of the chunk size
    client = chromadb.EphemeralClient()                      # in-memory database, nothing saved to disk
    col = client.create_collection(f"eval_{size}")           # a fresh collection for this size

    # Chunk every page and collect the text, labels, and unique IDs
    docs, metas, ids = [], [], []
    for filename, page_num, text in pages:
        for i, piece in enumerate(chunk_text(text, size, overlap)):
            docs.append(piece)
            metas.append({"source": filename, "page": page_num})
            ids.append(f"{filename}-{page_num}-{i}")

    # Turn all chunks into numbers and store them
    col.add(documents=docs, metadatas=metas, ids=ids,
            embeddings=model.encode(docs, show_progress_bar=False).tolist())

    hits_at_1 = 0    # right page was the very first result
    hits_at_k = 0    # right page was somewhere in the top 5
    for q in questions:
        # Search with this question
        res = col.query(query_embeddings=model.encode([q["question"]]).tolist(), n_results=TOP_K)
        found = [(m["source"], m["page"]) for m in res["metadatas"][0]]   # (file, page) of each result
        target = (q["source"], q["page"])                                  # the correct answer
        if found[0] == target:
            hits_at_1 += 1
        if target in found:
            hits_at_k += 1

    n = len(questions)
    return len(docs), hits_at_1 / n, hits_at_k / n


# Run the test for each chunk size and print a table
print(f"{len(questions)} test questions, top {TOP_K} results\n")
print(f"{'chunk size':>10} | {'chunks':>7} | {'top-1':>6} | {'top-' + str(TOP_K):>6}")
for size in CHUNK_SIZES:
    n_chunks, top1, topk = score(size)
    print(f"{size:>10} | {n_chunks:>7} | {top1:>5.0%} | {topk:>5.0%}")


# Show which questions failed at 200 words
client = chromadb.EphemeralClient()
col = client.create_collection("misses")
docs, metas, ids = [], [], []
for filename, page_num, text in pages:
    for i, piece in enumerate(chunk_text(text, 200, 40)):
        docs.append(piece); metas.append({"source": filename, "page": page_num}); ids.append(f"{filename}-{page_num}-{i}")
col.add(documents=docs, metadatas=metas, ids=ids, embeddings=model.encode(docs, show_progress_bar=False).tolist())
print("\nMissed at 200 words:")
for q in questions:
    res = col.query(query_embeddings=model.encode([q["question"]]).tolist(), n_results=TOP_K)
    found = [(m["source"], m["page"]) for m in res["metadatas"][0]]
    if (q["source"], q["page"]) not in found:
        print(f"- {q['question']}  (expected {q['source']} p{q['page']}; got {found})")