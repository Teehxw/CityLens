# THIS FILE TAKES A QUESTION AND FINDS THE MOST RELEVANT CHUNKS AND ASKS CLAUDE TO ANSWER FROM THEM
# import libraries
import os
import chromadb
from dotenv import load_dotenv  # this reads the API key from the .env file
from anthropic import Anthropic     # this is the CLAUDE API client
from sentence_transformers import SentenceTransformer   # turns text into numbers

# load the APi key from the .env file
load_dotenv()