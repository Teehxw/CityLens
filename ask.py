# THIS FILE TAKES A QUESTION AND FINDS THE MOST RELEVANT CHUNKS AND ASKS CLAUDE TO ANSWER FROM THEM
# import libraries
import os
import chromadb
from dotenv import load_dotenv  # this reads the API key from the .env file
from anthropic import Anthropic     # this is the CLAUDE API client
from sentence_transformers import SentenceTransformer   # turns text into numbers

# load the APi key from the .env file
load_dotenv()

# folder this file so the path works no matter where the script is run from
BASE = os.path.dirname(os.path.abspath(__file__))   # the CityLens folder

TOP_K = 5 # how many chunks to retrieve from the database fro each question
MODEL = "claude-sonnet-5-5" # the model to use for answering questions

# Load the same model that turns texts into numbers
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# open the database ingest.py created and get the collection of the chunks
collection = chromadb.PersistentClient(path=os.path.join(BASE, "chroma_db")).get_collection("citylens")

# establish a connection to the CLAUDE API
client = Anthropic()

# The rules we give clause for answering questions
SYSTEM_PROMPT = (
    "You answer questions about City of Edmonton Documents."
    "Use ONLY the numbered excerpts provided to answer the question."
    "If the excerpts do not contain the answer, reply exactly: "
    "\"I couldn't find the answer in the documents I have.\" "
    "Never guess or use outside knowledge."
    "Keep answers short and in plain language, and mention the source file and page you used"
)


def retrieve(question, k=TOP_K):
    """
    Given a question, retrieve the most relevant chunks from the database.
    """
    # turn the question into a vector of numbers
    question_vector = embedder.encode([question]).tolist()  # a list inside a list because the API expects a batch of questions

    # ask the database for the most relevant chunks
    results = collection.query(query_embeddings=question_vector, n_results=k)

    # make a list of the chunks and their metadata
    chunks = []
    # zip is going to walk through the texts and their labels (file and page) together
    for text, meta in zip(results['documents'][0], results["metadatas"][0]):
        chunks.append({"text": text, "source": meta["source"], "page": meta["page"]})

    return chunks


def answer(question):
    """
    Retrieve the chunks and ask Claude to answer the question from them
    """

    # step 1: retrieve the chunks
    chunks = retrieve(question)

    # step 2: build one block of text with the chubks numbered
    context = "\n\n".join(
        f"[{i+1}] ({c['source']}, page {c['page']})\n{c['text']}" for i, c in enumerate(chunks)
    )


    # step 3: send the rules, the pdfs, and the question to Claude
    message = client.messages.create(
        model=MODEL,
        max_tokens = 500,       # limit on how long the answer can be
        system = SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Excerpts:\n{context}\n\nQuestion: {question}"}],
    )

    #msg.content is a list of messages, we want the first one as it holds Claudes Answer
    return "".join(b.text for b in message.content if b.type == "text"), chunks

# This part only runs when you type "python3 ask.py".
# When another file (like app.py) imports ask.py, this part is skipped.
if __name__ == "__main__":
    q = input("Ask a question: ")      # wait for you to type a question
    text, chunks = answer(q)           # get Claude's answer and the chunks it was based on
    print("\n" + text)                 # print the answer
    print("\nSources retrieved:")      # then list where the chunks came from
    for c in chunks:
        print(f"  - {c['source']}, page {c['page']}")