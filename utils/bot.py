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
goods = ["Philips EP2231/40", "Nivona CafeRomatica NICR 550", # list of goods
        "Delonghi ECAM 370.70.B", "Polaris PACM 2065AC", 
        "Philips EP2030/10", "REDMOND RCM-1517"] 
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
    question = "Tell us about this coffee machine"
    template = """You are a useful AI consultant for our household appliances store selling coffee machines.
           Your task is to describe this coffee machine. Talk only about the merits.
           Use the following pieces of context (Context) to answer the question (Question) at the end.
           If you don't know the answer, just say you don't know, don't try to make up an answer.
           First, make sure the attached text is relevant to the question.
           If the question does not relate to the text, answer that you cannot answer this question.
           Use a maximum of 15 sentences.
           Give your answer as clearly as possible, briefly describing all the advantages of this particular coffee machine.
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
    template = """You are a useful AI consultant for our household appliances store selling coffee machines.
           Your task is to clearly answer the buyer's question.
           Use the following pieces of context (Context) to answer the question (Question) at the end.
           If you don't know the answer, just say you don't know, don't try to make up an answer.
           First, make sure the attached text is relevant to the question.
           If the question does not relate to the text, answer that you cannot answer this question.
           Use a maximum of 15 sentences.
           Make your answer as clear as possible. Speak competently.
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
    
def get_itemID(userID):
    cur.execute("SELECT itemID FROM user WHERE userID = ?", (userID,))
    fetch_result = cur.fetchone()
    return fetch_result[0]

@bot.message_handler(commands=["start"])
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
        cur.execute("UPDATE user SET status = ? WHERE userID = ?;", ("menu", message.chat.id))
    conn.commit()
    
@bot.message_handler(content_types="text", func=lambda message: check_step("menu", message.chat.id)) 
def machine_description(message):
    if message.text in goods:
        keyboard = types.ReplyKeyboardMarkup(
            resize_keyboard=True,
            one_time_keyboard=True
        )
        back_to_menu_button = types.KeyboardButton(text="Back to Menu")
        keyboard.add(back_to_menu_button)
        
        bot.send_message(message.chat.id, """Request accepted. Wait for a response...\nYou selected -> {}""".format(message.text))
        description = get_info(goods.index(message.text))
        bot.send_message(message.chat.id, description)
        bot.send_message(message.chat.id, """You can now ask questions about this product or return to the main menu to view another one.""", reply_markup=keyboard)
        # change user status in db
        cur.execute("UPDATE user SET status = ?, itemID = ?  WHERE userID = ?;", ("chat", 
                                                                                 goods.index(message.text), 
                                                                                 message.chat.id))
        conn.commit()
    else:
        bot.send_message(message.chat.id, "Request rejected. You entered wrong product name!")

@bot.message_handler(content_types="text", func= lambda message: check_step("chat", message.chat.id))
def chat_with_ai(message):
    keyboard = types.ReplyKeyboardMarkup(
            resize_keyboard=True,
            one_time_keyboard=True
        )
    back_to_menu_button = types.KeyboardButton(text="Back to Menu")
    keyboard.add(back_to_menu_button)
    
    if message.text == back_to_menu_button.text:
        bot.send_message(message.chat.id, "Returning back to Menu")
        cur.execute("UPDATE user SET status = ? WHERE userID = ?;", ("menu", message.chat.id))
        conn.commit()
        
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
        
    else:
        itemID = get_itemID(message.chat.id)
        answer = get_answer(itemID, message.text)
        bot.send_message(message.chat.id, answer, reply_markup=keyboard)
        
        
bot.infinity_polling(timeout=10, long_polling_timeout = 5)