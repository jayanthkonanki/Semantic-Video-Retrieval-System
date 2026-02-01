import chromadb 
import logging
import time
import random
import os 

def generate_time_based_id():
    current_time = time.localtime()
    formatted_time = time.strftime("Date: %Y-%m-%d  Time: %H:%M:%S", current_time)
    return formatted_time

def chromadb_setup(vector, summary,video_address):

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(module)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    dbpipeline = logging.getLogger('database_pipeline')
    dbpipeline.info('Setting up ChromaDB...')
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")

    client = chromadb.PersistentClient(path=CHROMA_PATH)
    dbpipeline.info('ChromaDB client created.')
    dbpipeline.info('Creating or getting collection...')
    collection = client.get_or_create_collection(name='VIDEOs')
    dbpipeline.info('Collection created or retrieved.')

    if not vector:
        dbpipeline.warning('No vectors provided. Exiting setup.')
        return
    
    dbpipeline.info('Inserting data into ChromaDB...')
    
    for i in range(len(vector)):
        collection.add(
            ids=[video_address[i]],
            embeddings=[vector[i]],
            documents=[summary[i]],
            metadatas=[{'upload_time': generate_time_based_id() }]  # Example metadata
        )

    dbpipeline.info('Data insertion complete.Total samples inserted: %s', len(vector))
    dbpipeline.info('ChromaDB setup complete.')
    return  
