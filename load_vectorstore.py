from qdrant_client import QdrantClient

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
api_key = 'Bt_er3Oh0Bes26noOnxCu11Kesy75cIbwxp36QWxnWEp8QCu8LkPfQ'
url = 'https://ca768691-8abf-47db-8f57-731619989033.us-east4-0.gcp.cloud.qdrant.io:6333'

openai_apikey = 'sk-proj-nQKp5FeRXsT8NKsTxE4WT3BlbkFJShIKU5x4aUY5CvBb5p10'

openai.api_key = openai_apikey

embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)


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

def load_word_document(doc):
    loader = UnstructuredWordDocumentLoader(
        doc, mode='single', strategy='fast'
    )
    docs = loader.load()
    return docs

"""
def load_pdf_document(doc):
    loader = UnstructuredPDFLoader(
        doc, mode='single', strategy='fast'
    )
    docs = loader.load()
    return docs

"""

data_folder = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'app_gpt/data')
document_route = os.path.join(data_folder, 'Manual-SQL1.docx')
#pdf_route = os.path.join(data_folder, 'AD620.pdf')


db = Qdrant.from_documents(
    url=url,
    documents=split_text(load_word_document(document_route)),
    embedding=OpenAIEmbeddings(openai_api_key=openai.api_key),
    api_key=api_key,
    collection_name='SQL-Manual'
)

"""
db = Qdrant.from_documents(
    url=url,
    documents=split_text(load_pdf_document(pdf_route)),
    embedding=OpenAIEmbeddings(openai_api_key=openai.api_key),
    api_key=api_key,
    collection_name='AD620'
)

"""



