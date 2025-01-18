import streamlit as st
import requests
from transformers import pipeline
from transformers import BlenderbotForConditionalGeneration, BlenderbotTokenizer
import os
from dotenv import load_dotenv
from datetime import datetime
import warnings
import google.generativeai as genai

# Suppress future warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.tokenization_utils_base")

# Load environment variables from .env file
load_dotenv()

# Cache the chatbot models to optimize performance
@st.cache_resource
def load_models():
    intent_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", revision="c626438")

    model_name = "facebook/blenderbot-400M-distill"
    chatbot_model = BlenderbotForConditionalGeneration.from_pretrained(model_name)
    chatbot_tokenizer = BlenderbotTokenizer.from_pretrained(model_name)
    return intent_classifier, chatbot_model, chatbot_tokenizer

intent_classifier, chatbot_model, chatbot_tokenizer = load_models()

# API keys
weather_api_key = os.getenv("WEATHER_API_KEY", "f07bdb36a61cde1e50acde6a8ab51d77")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyAPr3DkkQRsjdiCrNhEmptYQ8Fncf_Cs2s")
genai.configure(api_key=GOOGLE_API_KEY)

# Intent categories
intents = ["weather_query", "study_question", "chit_chat"]

# Chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Function to fetch Google search results
def google_search(query):
    try:
        model_name = None
        for model in genai.list_models():
            if 'generateContent' in model.supported_generation_methods:
                model_name = model.name
                break
        
        if not model_name:
            return "No suitable Generative AI model found for content generation."
        
        model = genai.GenerativeModel(model_name)
        chat = model.start_chat(history=[])
        response = chat.send_message(query, stream=True)
        return "".join(chunk.text for chunk in response if chunk.text)
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Function to get weather information
def get_weather(city):
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={weather_api_key}"
        response = requests.get(url)
        data = response.json()
        if response.status_code == 200:
            temp = data['main']['temp'] - 273.15
            description = data['weather'][0]['description']
            humidity = data['main']['humidity']
            wind_speed = data['wind']['speed']
            weather_info = f"Temperature: {temp:.2f}°C\nWeather: {description}\nHumidity: {humidity}%\nWind Speed: {wind_speed} m/s"
            return weather_info
        else:
            return f"Error: {data.get('message', 'Unable to fetch weather data.')}"
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Function to generate chat response using Blenderbot model
def generate_chat_response(user_input):
    inputs = chatbot_tokenizer.encode(user_input, return_tensors="pt")
    reply_ids = chatbot_model.generate(inputs)
    return chatbot_tokenizer.decode(reply_ids[0], skip_special_tokens=True)

# Function to classify intent
def classify_intent(user_input):
    result = intent_classifier(user_input, intents)
    return result['labels'][0]

# Streamlit interface
st.title("Multi-functional Chatbot with Intent Recognition")

user_input = st.text_input("Enter your query:")

if user_input:
    with st.spinner("Understanding your query..."):
        detected_intent = classify_intent(user_input)
    
    if detected_intent == "weather_query":
        st.write("Intent: Weather Query")
        city = user_input.replace("weather in", "").strip()
        with st.spinner("Fetching weather data..."):
            weather_info = get_weather(city)
        st.session_state.chat_history.append({"user": user_input, "bot": weather_info})
        st.write("**Weather Info:**", weather_info)
    
    elif detected_intent == "study_question":
        st.write("Intent: Study Question")
        with st.spinner("Finding the best answer..."):
            answer = google_search(user_input)
        st.session_state.chat_history.append({"user": user_input, "bot": answer})
        st.write("**Answer:**", answer)
    
    elif detected_intent == "chit_chat":
        st.write("Intent: Chit-chat")
        with st.spinner("Generating response..."):
            bot_reply = generate_chat_response(user_input)
        st.session_state.chat_history.append({"user": user_input, "bot": bot_reply})
        st.write("**Bot:**", bot_reply)

# Display chat history
st.write("### Chat History")
for message in st.session_state.chat_history:
    st.markdown(f"**You:** {message['user']}")
    st.markdown(f"**Bot:** {message['bot']}")
