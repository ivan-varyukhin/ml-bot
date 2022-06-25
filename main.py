from random import choice
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
import json
import random
import re
import nltk  # Natural Language Toolkit
import os
import telebot
from dotenv import load_dotenv

load_dotenv()


def filter_text(text):
    text = text.lower()
    pattern = r'[^\w\s]'
    text = re.sub(pattern, '', text)
    return text


def is_match(text1, text2):
    text1 = filter_text(text1)
    text2 = filter_text(text2)

    if len(text1) == 0 or len(text2) == 0:
        return False

    if text1.find(text2) != -1:
        return True

    if text2.find(text1) != -1:
        return True

    # Расстояние Левенштейна (расстояние редактирования)
    distance = nltk.edit_distance(text1, text2)
    length = (len(text1) + len(text2))/2
    score = distance / length

    return score < 0.8


# Конфигурация бота
BOT_CONFIG = {
    # Все намерения которые поддерживает наш бот
    'intents': {
        'hello': {
            'examples': ['Привет', 'Здарова', 'Йо', 'Приветос', 'Хеллоу'],
            'responses': ['Здравстсвтсвтвтвуй человек', 'И тебе не хворать', 'Здоровее видали'],
        },
        'how_are_you': {
            'examples': ['Как дела', 'Чо каво', 'Как поживаешь'],
            'responses': ['Маюсь Фигней', 'Веду интенсивы', 'Учу Пайтон'],
        }
    },
    # Фразы когда бот не может ответить
    'failure_phrases': ['Даже не знаю что сказать', 'Поставлен в тупик', 'Перефразируйте, я всего лишь бот'],
}

config_file = open('content/big_bot_config.json', 'r')
BIG_CONFIG = json.load(config_file)


X = []  # Фразы
y = []  # Намерения

# Собираем фразы и интенты из BIG_CONFIG в X,y
for name, intent in BIG_CONFIG['intents'].items():
    for example in intent['examples']:
        X.append(example)
        y.append(name)
    for example in intent['responses']:
        X.append(example)
        y.append(name)


vectorizer = CountVectorizer() 
vectorizer.fit(X)  


model = RandomForestClassifier()  # Выбираем ML модель
vecX = vectorizer.transform(X)  # Преобразуем тексты в вектора
model.fit(vecX, y)  # Обучаем модель
model.score(vecX, y)


def get_intent_ml(text):
    vec_text = vectorizer.transform([text])
    intent = model.predict(vec_text)[0]
    return intent


def get_intent(text):
    for name, intent in BOT_CONFIG['intents'].items():
        for example in intent['examples']:
            if is_match(text, example):
                return name
    return None


def ml_bot(phrase):
    phrase = filter_text(phrase)

    intent = get_intent(phrase)

    if not intent:
        intent = get_intent_ml(phrase)

    if intent:
        responses = BIG_CONFIG['intents'][intent]['responses']
        return choice(responses)

    failure = BIG_CONFIG['failure_phrases']
    return choice(failure)


BOT_KEY = os.environ.get('BOT_KEY')

bot = telebot.TeleBot(BOT_KEY)


@bot.message_handler(content_types=["text"])
def handle_text(message):
    bot.send_message(message.chat.id, ml_bot(message.text))


bot.polling()
