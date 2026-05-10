# This is the new, correct line
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from vector_utils import build_faiss_from_documents



# step 1:- load raw PDF
DATA_PATH = "data/"
def load_pdf_files(data_path=DATA_PATH):
    loader = DirectoryLoader(data_path,
                             glob="*.pdf",
                             loader_cls=PyPDFLoader)
    documents = loader.load()
    return documents

documents = load_pdf_files(data_path=DATA_PATH)
# print("Length of PDF pages: ",len(documents))

#create chunks
def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                   chunk_overlap=50,
                                                   separators=[". ", "\n\n", "\n", " ", ""])
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks = create_chunks(documents)
# print("Length of text chunks: ",len(text_chunks))

#create vector Embeddings

# Build and save FAISS index from the text chunks
db = build_faiss_from_documents(text_chunks)