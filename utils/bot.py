# import modules

from telebot import *
import logging
import sqlite3

# connect to the database
conn = sqlite3.connect(r"main.db", check_same_thread=False)
cur = conn.cursor()

# start logging
logging.basicConfig(level=logging.INFO, filename="../info.log", filemode='w')

# init a bot with token from file
bot_token_file = open(bot_token.txt, 'r')
bot_token = bot_token_file.readline()
bot_token_file.close()
bot = telebot.TeleBot(bot_token)

