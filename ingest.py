# Splits the documents into chunks and indexes them
import os   # for file path operations
from pypdf import PdfReader     # reads PDF files
from sentence_transformers import SentenceTransformer   # turns texts into numbers
import chromadb     # small database that stores the chunks

# Chunk size for splitting the documents
CHUNK_WORDS = 200
OVERLAP = 40

# Load the model that turns texts into numbers
model  = SentenceTransformer("all-MiniLM-L6-v2")

# Build paths from this file's location so the script works from any folder
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))   # the CityLens folder
DATA_DIR = os.path.join(PROJECT_DIR, "data")   # where the PDFs live
DB_DIR = os.path.join(PROJECT_DIR, "chroma_db")   # where the database is saved

# Create a ChromaDB client and collection
client = chromadb.PersistentClient(path=DB_DIR)   # store the database in the chroma_db folder
collection = client.get_or_create_collection("citylens")    # use if it exists, otherwise create a new collection

def chunk_text(text):
    """
    Cuts one page of text into overlapping chunks of 200 words each.
    """
    words = text.split()   # split the text into words
    step = CHUNK_WORDS - OVERLAP   # how many words to move forward for the next chunk
    for start in range(0, len(words), step):
        # take 200 words starting from the current position and join them back into a string
        piece  = " ".join(words[start:start + CHUNK_WORDS])
        yield piece  # return the chunk as a generator


# go through all the PDF files in the "data" folder
for filename in os.listdir(DATA_DIR):
    # ignore non-pdf Files
    if not filename.endswith(".pdf"):
        continue

    # read the PDF file
    reader = PdfReader(os.path.join(DATA_DIR, filename))

    # Three lists to hold the chunks, labels, and a unique ID for each chunk
    docs, metas, ids = [], [], []

    # Read the pdf one page at a time 
    for page_num, page in enumerate(reader.pages, start=1):
        # get the text of the page
        try:
            text = page.extract_text()
        except Exception:
            continue  # skip pages that can't be read
        if not text:  # skip pages with no text
            continue

        # split the page into chunks
        for i , piece in enumerate(chunk_text(text)):
            docs.append(piece)  # add the chunk to the list of documents
            metas.append({"source": filename, "page": page_num})   # add the metadata for the chunk
            ids.append(f"{filename}-page{page_num}-chunk{i}")   # unique ID for the chunk

    # turn the chunks into numbers
    if docs: # only if there are chunks to add
        embeddings = model.encode(docs)  # turn the chunks into numbers

        # save them 
        collection.upsert(documents = docs, metadatas = metas, ids = ids, embeddings = embeddings)

    print(f"{filename}: {len(docs)} chunks indexed")

# Show how many chunks are in the database total
print(f"Total chunks in database: {collection.count()}")