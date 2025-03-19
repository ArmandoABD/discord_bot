from pymongo import MongoClient
import os
from dotenv import load_dotenv
import logging
from nomic import embed
import json
import numpy as np
from PyPDF2 import PdfReader
from pathlib import Path

# Set up more detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pdf_setup.log')
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Nomic with API key
NOMIC_API_KEY = os.getenv('NOMIC_API_KEY')
if not NOMIC_API_KEY:
    logger.error("No Nomic API key found in .env file!")
    exit(1)

def setup_mongodb_pdf():
    """Connect to MongoDB and return the collection for AI documentation"""
    try:
        # Connect to MongoDB
        client = MongoClient(os.getenv('MONGODB_URI'))
        db = client['ai_documentation']  # New database for AI documentation
        collection = db['model_docs']    # New collection for model documentation
        
        # Check for vector search indexes
        indexes = list(collection.list_indexes())
        vector_indexes = [index['name'] for index in indexes]
        logger.info(f"Found indexes in collection: {', '.join(vector_indexes)}")
        
        if 'discord_bot_ai' in vector_indexes:
            logger.info("Vector search index 'discord_bot_ai' is ready for use")
        
        logger.info(f"Successfully connected to MongoDB collection: {collection.name}")
        logger.info(f"Current document count: {collection.count_documents({})}")
        return collection
    except Exception as e:
        logger.error(f"Error connecting to MongoDB: {str(e)}")
        raise

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file"""
    try:
        reader = PdfReader(pdf_path)
        logger.info(f"Processing PDF: {pdf_path}")
        logger.info(f"Number of pages: {len(reader.pages)}")
        
        text = ""
        for i, page in enumerate(reader.pages, 1):
            page_text = page.extract_text()
            text += page_text + "\n"
            logger.info(f"Extracted page {i}: {len(page_text)} characters")
        
        logger.info(f"Total extracted text length: {len(text)} characters")
        if len(text) == 0:
            logger.warning(f"Extracted text is empty for {pdf_path}, skipping...")
            return None
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF {pdf_path}: {str(e)}")
        return None

def chunk_text(text, chunk_size=1000, overlap=100):
    """Split text into overlapping chunks"""
    logger.info(f"Chunking text of length {len(text)}")
    chunks = []
    if not text:
        logger.warning("Empty text provided to chunking function")
        return chunks
    
    try:
        start = 0
        chunk_count = 0
        last_start = -1  # Track the last start position to detect infinite loops
        
        # Safety counter to prevent infinite loops
        max_iterations = len(text) * 2  # Should never need more iterations than twice the text length
        iterations = 0
        
        while start < len(text) and iterations < max_iterations:
            iterations += 1
            
            # Detect infinite loop - if we're processing the same start position again
            if start == last_start:
                logger.error(f"Infinite loop detected at position {start}, breaking chunking process")
                break
                
            last_start = start
            
            end = min(start + chunk_size, len(text))
            chunk = text[start:end]
            
            # If not at the end, try to break at a sentence or paragraph
            if end < len(text):
                # Try to find a good breaking point (period followed by space)
                break_point = chunk.rfind('. ')
                if break_point != -1:
                    end = start + break_point + 2  # Include the period and space
                    chunk = text[start:end]
            
            chunks.append(chunk)
            chunk_count += 1
            
            # Critical fix: ensure we're making forward progress
            new_start = end - overlap
            if new_start <= start:
                logger.warning(f"No forward progress at position {start}, adjusting to continue")
                # Force progress by moving forward at least one character
                new_start = start + 1
            
            start = new_start
        
        # Check if we hit the iteration limit
        if iterations >= max_iterations:
            logger.error(f"Chunking aborted after {iterations} iterations - possible infinite loop")
        
        logger.info(f"Created {len(chunks)} chunks")
        return chunks
    except Exception as e:
        logger.error(f"Error during chunking: {str(e)}")
        return chunks

def process_and_upload_documents(files_dir, collection):
    """Process PDF documents and upload them to MongoDB with embeddings"""
    # Get all PDF files in the directory
    pdf_files = list(Path(files_dir).glob('*.pdf'))
    
    if not pdf_files:
        logger.info(f"No PDF files found in {files_dir}")
        return
    
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    total_chunks_processed = 0
    
    for i, pdf_path in enumerate(pdf_files):
        try:
            logger.info(f"Processing file {i+1}/{len(pdf_files)}: {pdf_path}")
            
            # Extract text from PDF
            text = extract_text_from_pdf(pdf_path)
            if not text:
                logger.warning(f"No text extracted from {pdf_path}, skipping...")
                continue
                
            # Create chunks with overlap
            chunks = chunk_text(text)
            if not chunks:
                logger.warning(f"No chunks created from {pdf_path}, skipping...")
                continue
            
            # Generate embeddings for all chunks 
            logger.info(f"Generating embeddings for {len(chunks)} chunks...")
            try:
                # Split larger documents into smaller batches for embedding
                max_batch_size = 2  # Small batches to avoid rate limits
                all_embeddings = []
                
                for batch_start in range(0, len(chunks), max_batch_size):
                    batch_end = min(batch_start + max_batch_size, len(chunks))
                    batch = chunks[batch_start:batch_end]
                    
                    # Implement retry logic with increasing delays
                    max_retries = 5
                    retry_count = 0
                    retry_delay = 1
                    
                    while retry_count < max_retries:
                        try:
                            # Add a small delay between batches to avoid rate limiting
                            if batch_start > 0:
                                import time
                                time.sleep(2)  # 2-second delay between API calls
                                
                            output = embed.text(
                                texts=batch,
                                model='nomic-embed-text-v1.5',
                                task_type='search_document',
                                dimensionality=384  # Match MongoDB index dimensions
                            )
                            
                            batch_embeddings = output['embeddings']
                            all_embeddings.extend(batch_embeddings)
                            break  # Break out of retry loop on success
                            
                        except Exception as e:
                            retry_count += 1
                            if retry_count >= max_retries:
                                logger.error(f"Failed to process batch after {max_retries} retries: {str(e)}")
                                raise  # Re-raise to be caught by outer try/except
                            
                            # Exponential backoff
                            logger.warning(f"API error, retry {retry_count}/{max_retries} after {retry_delay}s: {str(e)}")
                            import time
                            time.sleep(retry_delay)
                            retry_delay *= 2  # Double the delay for next retry
                
                embeddings = all_embeddings
                    
                if not embeddings:
                    logger.warning(f"No embeddings generated for {pdf_path}, skipping...")
                    continue
                logger.info(f"Generated {len(embeddings)} embeddings")
            except Exception as e:
                logger.error(f"Error generating embeddings for {pdf_path}: {str(e)}")
                continue
            
            # Upload documents with their embeddings
            logger.info("Uploading to MongoDB...")
            for j, (chunk, embedding) in enumerate(zip(chunks, embeddings), 1):
                # Create document
                doc = {
                    "content": chunk,
                    "embedding": embedding,
                    "metadata": {
                        "source_file": str(pdf_path),
                        "file_name": pdf_path.name,
                        "chunk_size": len(chunk),
                        "chunk_number": j,
                        "total_chunks": len(chunks)
                    }
                }
                
                # Insert into MongoDB
                try:
                    result = collection.insert_one(doc)
                    total_chunks_processed += 1
                except Exception as e:
                    logger.error(f"Error uploading chunk {j} from {pdf_path}: {str(e)}")
                
            logger.info(f"Successfully processed and uploaded {pdf_path}")
                
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {str(e)}")
    
    logger.info(f"Processing complete. Total chunks processed: {total_chunks_processed}")
    logger.info(f"Final document count in collection: {collection.count_documents({})}")

if __name__ == "__main__":
    # Connect to MongoDB
    collection = setup_mongodb_pdf()
    
    # Process files in the 'files' directory
    files_dir = "files"
    if not os.path.exists(files_dir):
        os.makedirs(files_dir)
        logger.info(f"Created files directory at {files_dir}")
    
    # Process and upload documents
    process_and_upload_documents(files_dir, collection)
    logger.info("Document processing complete") 