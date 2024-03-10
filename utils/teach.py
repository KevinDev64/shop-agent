#!/bin/python3

import os
import langchain
from pprint import pprint as pp
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import warnings

token_file = open("openai_token.txt", "r")
token = token_file.readline()
token_file.close()

os.environ["OPENAI_API_KEY"] = token

number_of_goods = 6 # fix this

print("-------------------------------- splitting")
data_plus = []
path = '../input/'
text_loader_kwargs={'autodetect_encoding': True}
splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=10,
    separators=["\n\n", "\n", "(?<=\. )"]
)
for i in range(number_of_goods):
    loader = DirectoryLoader(path, glob="text"+str(i)+".txt", loader_cls=TextLoader,
                             loader_kwargs=text_loader_kwargs)
    docs = loader.load()
    
    data_plus.append(splitter.split_documents(docs))
    
pp(data_plus)
input("Waiting for Enter to continue...")
print("-------------------------------- translate to vectors & save to files...")
directory = ''
embedding = OpenAIEmbeddings()
vectordb_list = []
for i in range(number_of_goods):
    vectordb_list.append(Chroma.from_documents(documents=data_plus[i], 
                                               embedding=embedding, persist_directory="../output/"+str(i)))
for j in range(number_of_goods):
    vectordb_list[j].persist()
    print(vectordb_list[j]._collection.count())