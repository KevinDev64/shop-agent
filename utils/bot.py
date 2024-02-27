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
