# import modules

import telebot
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
API_KEY = bot_token_file.readline()
bot_token_file.close()
bot = telebot.TeleBot("7174085128:AAGfMlZh5wUoV3vXfoGOYtb9vkN3SbqOmAE")

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
def check_step(step, id): 
    cur.execute("SELECT status FROM user WHERE userID = ?", (id,))
    fetch_result = cur.fetchone()
    if step in fetch_result:
        return True
    else:
        return False
        
print("Init bot decorators")

@bot.message_handler(commands=["start", "старт", "начать"])
def start_message(message):
    keyboard = types.ReplyKeyboardMarkup(
        resize_keyboard = True,
        one_time_keyboard=True
    )
    zero_machine = types.KeyboardButton(text="Philips EP2231/40")
    first_machine = types.KeyboardButton(text="Nivona CafeRomatica NICR 550")
    second_machine = types.KeyboardButton(text="Delonghi ECAM 370.70.B")
    third_machine = types.KeyboardButton(text="Polaris PACM 2065AC")
    fourth_machine = types.KeyboardButton(text="Philips EP2030/10")
    fifth_machine = types.KeyboardButton(text="REDMOND RCM-1517")
    
    keyboard.row(zero_machine, first_machine)
    keyboard.row(second_machine, third_machine)
    keyboard.row(fourth_machine, fifth_machine)
    bot.send_message(message.chat.id, "Main menu", reply_markup=keyboard)
    
    try:
        cur.execute("INSERT INTO user VALUES (?, ?, ?);", (message.chat.id, "menu", 0))
    except:
        cur.execute("UPDATE user SET status = ? WHERE userID = ?;", ("chat", message.chat.id))
    conn.commit()
    
@bot.message_handler(content_types="text", func=lambda message: message.text == "Philips EP2231/40" and check_step("menu", message.chat.id)) 
def zero_machine(message):
    pass
    
bot.polling() 