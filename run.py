import time 
import os 
from preprocessing import upload_your_dataset
from transformers import BlipProcessor, BlipForQuestionAnswering #type: ignore
from sentence_transformers import SentenceTransformer #type: ignore
import logging
import chromadb  #type: ignore
import streamlit as st  #type: ignore


def init_worker_VQA():
    global video_captioning_processor
    global video_captioning_model
    video_captioning_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base", use_fast=True)
    video_captioning_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

def init_worker_text_vectorization():
    global text_vectorization_model
    text_vectorization_model = SentenceTransformer("all-MiniLM-L6-v2")

def folder_walkthrough(root_folder):
    data_paths = []
    for dirpath, _, filenames in os.walk(root_folder):
        for fname in filenames:
            data_paths.append(os.path.join(dirpath, fname))
    return data_paths


# ------------------- QUERY FUNCTION -------------------
def QUERY(Query):
  global text_vectorization_model
  ChromaDB_Query_Embeddings = text_vectorization_model.encode(Query, convert_to_numpy=True)
  ChromaDB_Query_result = chromadb_collection.query(query_embeddings = ChromaDB_Query_Embeddings,
                 n_results=1)
  st.write(f"using chromaDB: Your query is related to the document is at :{ChromaDB_Query_result['ids'][0][0]}" )

# ------------------- Safe Main Runner -------------------

if __name__ == '__main__':
    
    st.title('WELCOME TO VIDEO-RETRIVEAL SYSTEM PROTOTYPE')
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CHROMA_PATH = os.path.join(BASE_DIR, "chroma_db")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    chromadb_collection = client.get_or_create_collection(name = 'VIDEOs')

    streamlit_col1,streamlit_col2 = st.columns(2)

    logging.basicConfig(level=logging.INFO,format='%(asctime)s - %(levelname)s - %(message)s',datefmt='%Y-%m-%d %H:%M:%S')
    pipeline = logging.getLogger('pipeline')
    pipeline.info('Starting the execution...')
    tic = time.time()
    pipeline.info('Initializing workers...')
    init_worker_VQA()
    init_worker_text_vectorization()
    pipeline.info('Initialization complete.')
    pipeline.info('Getting video addresses...') 
    
    video_address = []

    def save_uploaded_file(uploaded_files):
           BASE_DIR = os.path.dirname(os.path.abspath(__file__))
           folder = os.path.join(BASE_DIR, "storage")
           os.makedirs(folder, exist_ok=True)
           file_path = os.path.join(folder, uploaded_files.name)
           with open(file_path, "wb") as f:
               f.write(uploaded_files.getbuffer())  # Efficient and clean
           return file_path
    if 'all_uploaded_files' not in st.session_state:
        st.session_state.all_uploaded_files = set()
    if 'new_uploaded_files' not in st.session_state:
        st.session_state.new_uploaded_files = []

    with streamlit_col1: 
        uploaded_files = st.file_uploader('Upload your video files here:', type=['mp4'], accept_multiple_files=True)    

        if uploaded_files:
           
           st.session_state.new_uploaded_files = [file for file in uploaded_files if file.name not in st.session_state.all_uploaded_files]
           if not st.session_state.new_uploaded_files:
                st.warning("No new files to upload to database.")

           elif st.button('🚀 Upload & Process'): 
              
              for files in st.session_state.new_uploaded_files:
                 video_address.append(save_uploaded_file(files))

              pipeline.info('no of videos:%s', len(video_address))
              pipeline.info('Video addresses retrieved.')
              pipeline.info('Uploading dataset...')

              upload_your_dataset(video_address)

              pipeline.info('Dataset uploaded successfully.')
              toc = time.time()
              pipeline.info('Time taken to upload the dataset: %s seconds', toc - tic)
              pipeline.info('DATASET_UPLOADED...')
              for files in st.session_state.new_uploaded_files:
                 st.session_state.all_uploaded_files.add(files.name)
              st.success(f"Processed new files!")
    
    #video_address = folder_walkthrough(dataset_path)
    def QUERY(Query,chromadb_collection):
       global text_vectorization_model
       ChromaDB_Query_Embeddings = text_vectorization_model.encode(Query, convert_to_numpy=True)
       ChromaDB_Query_result = chromadb_collection.query(query_embeddings = ChromaDB_Query_Embeddings,
                 n_results=1)
       st.write(f"using chromaDB: Your query is related to the document is at :{ChromaDB_Query_result['ids'][0][0]}" )
       st.success('Query submitted successfully.')
    

    with streamlit_col2:
        
        if chromadb_collection.count():
            if "query" not in st.session_state:
                st.session_state.query = ""
            Query = st.text_input('Enter your query: ')
            if Query:
               if st.button('Submit'):
                  QUERY(Query,chromadb_collection)
                  pipeline.info('Exiting the program.')  
        else:
            st.write("Insufficient data! Upload your data for querying")
            
            

