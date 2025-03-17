from pymongo import MongoClient
import os
from dotenv import load_dotenv
import logging
from nomic import embed
import json
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Nomic with API key
NOMIC_API_KEY = os.getenv('NOMIC_API_KEY')
if not NOMIC_API_KEY:
    logger.error("No Nomic API key found in .env file!")
    exit(1)

def setup_mongodb():
    """Connect to MongoDB and return the collection"""
    try:
        # Connect to MongoDB
        client = MongoClient(os.getenv('MONGODB_URI'))
        db = client[os.getenv('MONGODB_DB', 'sample_mflix')]
        collection = db[os.getenv('MONGODB_COLLECTION', 'movies')]
        logger.info(f"Successfully connected to MongoDB collection: {collection.name}")
        return collection
    except Exception as e:
        logger.error(f"Error connecting to MongoDB: {str(e)}")
        raise

def process_and_upload_documents(file_paths, collection):
    """Process documents and upload them to MongoDB with embeddings"""
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                
                # Create chunks (simple splitting for now, you might want to use a more sophisticated chunking strategy)
                chunks = [content[i:i+1000] for i in range(0, len(content), 1000)]
                
                # Generate embeddings for all chunks at once
                output = embed.text(
                    texts=chunks,
                    model='nomic-embed-text-v1.5',
                    task_type='search_document',
                    dimensionality=384  # Match MongoDB index dimensions
                )
                embeddings = output['embeddings']
                
                # Upload documents with their embeddings
                for chunk, embedding in zip(chunks, embeddings):
                    # Create document
                    doc = {
                        "content": chunk,
                        "embedding": embedding,
                        "metadata": {
                            "source": file_path,
                            "chunk_size": len(chunk)
                        }
                    }
                    
                    # Insert into MongoDB
                    collection.insert_one(doc)
                    
                logger.info(f"Processed and uploaded {file_path}")
                
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")

if __name__ == "__main__":
    # Just test the connection
    collection = setup_mongodb()
    logger.info("MongoDB connection test successful")

    # Example usage:
    # Specify the directory containing your documents
    docs_dir = "documents"
    if not os.path.exists(docs_dir):
        os.makedirs(docs_dir)
        logger.info(f"Created documents directory at {docs_dir}")
    
    # Get all files in the documents directory
    file_paths = [os.path.join(docs_dir, f) for f in os.listdir(docs_dir) if os.path.isfile(os.path.join(docs_dir, f))]
    
    if file_paths:
        process_and_upload_documents(file_paths, collection)
        logger.info("Document processing complete")
    else:
        logger.info("No documents found in the documents directory") 