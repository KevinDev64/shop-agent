# import modules

from telebot import *
import logging
import sqlite3
import os
import langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.document_loaders import DirectoryLoader
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# connect to the database
conn = sqlite3.connect(r"main.db", check_same_thread=False)
cur = conn.cursor()

# start logging
logging.basicConfig(level=logging.INFO, filename="../info.log", filemode='w')

# init a bot with token from file
bot_token_file = open("bot_token.txt", "r")
bot_token = bot_token_file.readline()
bot_token_file.close()
bot = telebot.TeleBot(bot_token)

# set the openai token
token_file = open("openai_token.txt", "r")
token = token_file.readline()
token_file.close()
os.environ["OPENAI_API_KEY"] = token

docs_k = 65 # const
number_of_goods = 6 # const
langchain.debug = False # debug is off

# read the vector databases
vectordb_list = []
embedding = OpenAIEmbeddings()
for i in range(number_of_goods):
    vectordb_list.append(Chroma(embedding_function=embedding, 
                                persist_directory="../output/"+str(i)))
for vectordb in vectordb_list:
    print(vectordb._collection.count())
    
def get_info(itemID):
    question = "Расскажи про эту кофемашину"
    template = """Ты - полезный ИИ консультант для нашего магазина бытовой техники по продаже кофемашин.
        Твое задание - описать данную кофемашину. Говори только о достоинствах.
        Используйте следующие фрагменты контекста (Context), чтобы ответить на вопрос в конце (Question).
        Если вы не знаете ответа, просто скажите, что не знаете, не пытайтесь придумывать ответ.
        Сначала убедитесь, что прикрепленный текст имеет отношение к вопросу.
        Если вопрос не имеет отшения к тексту, ответьте, что вы не можете ответить на данный вопрос.
        Используйте максимум 15 предложений. 
        Дайте ответ как можно более понятным, рассказывая кратко про все достинства именно данной кофемашины.
        Context: {context}
        Question: {question}""" 
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
    
    vectordb = vectordb_list[itemID]
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": docs_k})
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0,
        max_tokens = 250)
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})
    result = qa_chain({"query": question})
    return result["result"]

def get_answer(itemID, question):
    template = """Ты - полезный ИИ консультант для нашего магазина бытовой техники по продаже кофемашин.
        Твое задание - понятно ответить на вопрос покупателя. 
        Используйте следующие фрагменты контекста (Context), чтобы ответить на вопрос в конце (Question).
        Если вы не знаете ответа, просто скажите, что не знаете, не пытайтесь придумывать ответ.
        Сначала убедитесь, что прикрепленный текст имеет отношение к вопросу.
        Если вопрос не имеет отшения к тексту, ответьте, что вы не можете ответить на данный вопрос.
        Используйте максимум 15 предложений. 
        Дайте ответ как можно более понятным. Говорите грамотно.
        Context: {context}
        Question: {question}""" 
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
    
    vectordb = vectordb_list[itemID]
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": docs_k})
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0,
        max_tokens = 250)
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})
    result = qa_chain({"query": question})
    return result["result"]