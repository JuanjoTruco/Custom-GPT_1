# Description: This file is used to load the vector store and the document loader for the GPT-3 model.
# the vector store is used to store the embeddings of the documents and the document loader is used to load the documents
# and split them into smaller chunks for processing by the GPT-3 model.
# 
# By: Juanjo Escobar, special thanks to my teacher Alejandro Puerta E.
#
# --------------------------------------------------------------------------------------------------------------------------------------- #


import openai
# from langchain.document_loaders import UnstructuredWordDocumentLoader
#from langchain.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings

from langchain_community.vectorstores import Qdrant
from langchain.chains import RetrievalQA
import os
from langchain_community.output_parsers.rail_parser import GuardrailsOutputParser


##vector store API Key and URL
api_key = None #your api key
url = None #your url

openai_apikey = None #your openai api key
openai.api_key = openai_apikey

embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)

# Function to split the text into smaller chunks
def split_text(doc):
    text_content = doc[0].page_content
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 20,
        length_function = len,
        add_start_index = True
    )
    texts = text_splitter.create_documents([text_content])
    return texts

# Function to load the word document
def load_word_document(doc):
    loader = UnstructuredWordDocumentLoader(
        doc, mode='single', strategy='fast'
    )
    docs = loader.load()
    return docs

# Function to load the pdf document
"""
def load_pdf_document(doc):
    loader = UnstructuredPDFLoader(
        doc, mode='single', strategy='fast'
    )
    docs = loader.load()
    return docs

"""
# Path to the data folder
data_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'your_data_folder')
document_route = os.path.join(data_folder, 'Your_document.docx'))
#pdf_route = os.path.join(data_folder, 'AD620.pdf')

# Load the vector store
db = Qdrant.from_documents(
    url=url,
    documents=split_text(load_word_document(document_route)),
    embedding=OpenAIEmbeddings(openai_api_key=openai.api_key),
    api_key=api_key,
    collection_name='Your_collection_name'
)

# Load the vector store in case you use a pdf document
"""
db = Qdrant.from_documents(
    url=url,
    documents=split_text(load_pdf_document(pdf_route)),
    embedding=OpenAIEmbeddings(openai_api_key=openai.api_key),
    api_key=api_key,
    collection_name='AD620'
)

"""



