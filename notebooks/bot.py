import logging
from aiogram import Bot, Dispatcher, types
from aiogram.types import ParseMode
from aiogram.contrib.middlewares.logging import LoggingMiddleware
from aiogram.dispatcher import Dispatcher
from aiogram.dispatcher.filters import Text
from aiogram.utils import executor
from deep_translator import GoogleTranslator
import joblib

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.probability import FreqDist
from nltk.util import bigrams, ngrams

import re
import string
from deep_translator import GoogleTranslator

from source import model
logging.basicConfig(level=logging.INFO)



mmodel = model()
TOKEN = '7331126380:AAHntnAi8bAb1mZC1ZWVKeHmaSDYva7adYM'
bot = Bot(token=TOKEN)
dp = Dispatcher(bot)
dp.middleware.setup(LoggingMiddleware())
stop_words = set(stopwords.words('english') + list(string.ascii_lowercase))
en_stops = set(stopwords.words('english'))
lemmatizer  = WordNetLemmatizer()


def delete_stop_words_lemmatization_punctiation(row):
    row = re.sub(r"\n", "", row.lower())
    row = re.sub(r"[^\w\s]", ' ', row)
    row_list = row.split(' ')
    row_list_withut_stops = [word for word in row_list if word not in en_stops]
    text = [lemmatizer.lemmatize(w) for w in row_list_withut_stops]
    return ' '.join(text)


@dp.message_handler(commands=['start'])
async def send_welcome(message: types.Message):
    """Отправляет сообщение, когда команда /start выдана."""
    await message.reply("Привет! Отправь мне сообщение, и я обработаю его.")


@dp.message_handler(Text)
async def handle_message(message: types.Message):
    """Обрабатывает входящие сообщения от пользователей."""
    user_message = message.text
    translated_text = GoogleTranslator(source='auto', target='en').translate(user_message)
    cleaned_message = delete_stop_words_lemmatization_punctiation(translated_text)
    prediction = mmodel.predict(str(cleaned_message))
    await message.reply(f'Предсказание: {prediction}')


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
