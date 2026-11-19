# ask.py: takes a question, finds the most relevant chunks, and asks Claude to answer from them
import os                                                   # for building file paths
import chromadb                                             # the database that holds our chunks
from dotenv import load_dotenv                              # reads the API key from the .env file
from anthropic import Anthropic                             # lets us talk to Claude
from sentence_transformers import SentenceTransformer       # turns text into numbers

load_dotenv()  # loads ANTHROPIC_API_KEY from .env so the Anthropic client can find it

# Folder this file lives in (the CityLens folder), so paths work no matter where you run from
BASE = os.path.dirname(os.path.abspath(__file__))

TOP_K = 5                    # how many chunks to retrieve for each question
MODEL = "claude-sonnet-5-5"  # which Claude model answers (you can change this later)

# Load the same number-making model that ingest.py used.
# It must be the same one, so questions and chunks are measured the same way.
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Open the database that ingest.py created and get the collection of chunks
collection = chromadb.PersistentClient(path=os.path.join(BASE, "chroma_db")).get_collection("citylens")

# Create the connection to Claude (it uses the API key loaded above)
client = Anthropic()

# The rules we give Claude. This is what keeps it from making things up.
SYSTEM = (
    "You answer questions about City of Edmonton documents. "
    "Use ONLY the numbered excerpts provided. "                  # only use what we give you
    "If the excerpts do not contain the answer, reply exactly: "
    "\"I couldn't find that in the documents I have.\" "         # the "I don't know" reply
    "Never guess or use outside knowledge. "
    "Keep answers short and in plain language, and mention the source file and page you used."
)


def retrieve(question, k=TOP_K):
    """Find the k stored chunks closest in meaning to the question."""
    # Turn the question into numbers (a list inside a list, because the database expects a batch)
    q_vec = embedder.encode([question]).tolist()

    # Ask the database for the k chunks whose numbers are closest to the question's numbers
    res = collection.query(query_embeddings=q_vec, n_results=k)

    chunks = []
    # The results come back in lists, one entry per question. We asked one question, so we use [0].
    # zip walks through the texts and their labels (file and page) together.
    for text, meta in zip(res["documents"][0], res["metadatas"][0]):
        chunks.append({"text": text, "source": meta["source"], "page": meta["page"]})
    return chunks


def answer(question):
    """Retrieve chunks, ask Claude, and return (answer_text, chunks)."""
    chunks = retrieve(question)   # step 1: find the relevant chunks

    # Step 2: build one block of text with the chunks numbered, e.g.
    # [1] (zoning_guide.pdf, page 12)
    # ...chunk text...
    context = "\n\n".join(
        f"[{i+1}] ({c['source']}, page {c['page']})\n{c['text']}" for i, c in enumerate(chunks)
    )

    # Step 3: send the rules, the excerpts, and the question to Claude
    message = client.messages.create(
        model=MODEL,
        max_tokens=1500,       # limit on how long the reply can be (room for thinking + answer)
        system=SYSTEM,         # the rules from above
        messages=[{"role": "user", "content": f"Excerpts:\n{context}\n\nQuestion: {question}"}],
    )

    # Claude's reply is a list of blocks. Some are "thinking" blocks (no .text),
    # so we keep only the blocks of type "text" and join them into the answer.
    answer_text = "".join(block.text for block in message.content if block.type == "text")
    return answer_text, chunks


# This part only runs when you type "python3 ask.py".
# When another file (like app.py) imports ask.py, this part is skipped.
if __name__ == "__main__":
    q = input("Ask a question: ")      # wait for you to type a question
    text, chunks = answer(q)           # get Claude's answer and the chunks it was based on
    print("\n" + text)                 # print the answer
    print("\nSources retrieved:")      # then list where the chunks came from
    for c in chunks:
        print(f"  - {c['source']}, page {c['page']}")