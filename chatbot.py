import nltk
import random
import json
import string
import datetime
import wikipedia
import requests
import warnings
from nltk.stem import WordNetLemmatizer
from bs4 import GuessedAtParserWarning

# Suppress parser warnings
warnings.filterwarnings("ignore", category=GuessedAtParserWarning)

# NLTK data setup
nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# Load intents
with open("intents_better.json", "r") as file:
    data = json.load(file)

# Weather API key (replace with your actual key)
WEATHER_API_KEY = "eea19048fad34faca85102630250507"

def preprocess(sentence):
    tokens = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(word.lower()) for word in tokens if word not in string.punctuation]

def match_intent(user_input):
    input_tokens = preprocess(user_input)
    best_tag = None
    best_response = "Sorry, I don't understand that."
    max_overlap = 0

    for intent in data['intents']:
        for pattern in intent['patterns']:
            pattern_tokens = preprocess(pattern)
            overlap = len(set(pattern_tokens).intersection(set(input_tokens)))
            if overlap > max_overlap:
                max_overlap = overlap
                best_tag = intent['tag']
                best_response = random.choice(intent['responses'])

                # Personalize identity intent
                if best_tag == "identity":
                    name = next((w for w in user_input.split() if w.istitle()), None)
                    if name:
                        best_response = best_response.replace("{name}", name)

    return best_tag, best_response

def extract_city_from_input(text):
    tokens = text.lower().split()
    if "in" in tokens:
        idx = tokens.index("in")
        if idx + 1 < len(tokens):
            return tokens[idx + 1].capitalize()
    return None

def get_weather(city):
    url = f"http://api.weatherapi.com/v1/forecast.json?key={WEATHER_API_KEY}&q={city}&days=1&aqi=yes&alerts=no"
    try:
        res = requests.get(url)
        data = res.json()
        if "error" in data:
            return f"City '{city}' not found. Try another."

        location = data["location"]["name"]
        country = data["location"]["country"]
        condition = data["current"]["condition"]["text"]
        temp = data["current"]["temp_c"]
        feels = data["current"]["feelslike_c"]
        humidity = data["current"]["humidity"]
        wind_kph = data["current"]["wind_kph"]

        return (
            f"The weather in {location}, {country} is currently '{condition}' "
            f"with a temperature of {temp}°C (feels like {feels}°C), "
            f"humidity at {humidity}%, and wind speed {wind_kph} kph."
        )
    except Exception as e:
        return f"Error retrieving weather: {str(e)}"

def get_real_time_response(tag, user_input):
    if tag == "time":
        return "The current time is " + datetime.datetime.now().strftime("%H:%M:%S")

    elif tag == "weather":
        city = extract_city_from_input(user_input)
        if not city:
            return "Please specify a city, like 'weather in London'."
        return get_weather(city)

    elif "wikipedia" in user_input.lower() or user_input.lower().startswith("wiki"):
        try:
            topic = user_input.lower().replace("wikipedia", "").replace("wiki", "").strip()
            if not topic:
                return "Please specify a topic after 'Wikipedia' or 'wiki'."
            
            # Try fetching the topic directly
            summary = wikipedia.summary(topic, sentences=2)
            return summary

        except wikipedia.exceptions.DisambiguationError as e:
            topic_cap = topic.capitalize()

            # Try forcing the exact topic if it's in the options
            if topic_cap in e.options:
                try:
                    summary = wikipedia.summary(topic_cap, sentences=2)
                    return f"Here's some info on '{topic_cap}': {summary}"
                except Exception:
                    pass

            # Show user some disambiguation options
            top_options = ", ".join(e.options[:5])
            return f"The topic '{topic}' is broad. Please choose one of these: {top_options}"

        except wikipedia.exceptions.PageError:
            return f"I couldn't find a page for '{topic}'. Try another term."

        except Exception as e:
            return f"An error occurred while fetching Wikipedia info: {str(e)}"

    return None


def chat():
    print("Bot: Hello! Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['quit', 'exit']:
            print("Bot: Goodbye!")
            break
        tag, base_response = match_intent(user_input)
        real_time_response = get_real_time_response(tag, user_input)
        print(f"Bot: {real_time_response if real_time_response else base_response}")

if __name__ == "__main__":
    chat()
