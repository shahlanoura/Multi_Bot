import streamlit as st
import requests
from transformers import BlenderbotForConditionalGeneration, BlenderbotTokenizer
import os
from dotenv import load_dotenv
from datetime import datetime
import warnings

warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.tokenization_utils_base")

# Load environment variables from .env file
load_dotenv()

# Load the fine-tuned model and tokenizer for Blenderbot
@st.cache_resource
def load_model():
    # Load your model here
    return BlenderbotForConditionalGeneration.from_pretrained("facebook/blenderbot-400M-distill")

model_name = "facebook/blenderbot-400M-distill"
model = BlenderbotForConditionalGeneration.from_pretrained(model_name)
tokenizer = BlenderbotTokenizer.from_pretrained(model_name)

# Get the API key for weather API from environment variables
weather_api_key = os.getenv("API_KEY", "f07bdb36a61cde1e50acde6a8ab51d77") 

# Set up Google Custom Search API settings
GOOGLE_API_KEY = "AIzaSyBcvvpvj1EPtxwhYTaZCctLC76O5_nlqBA"
GOOGLE_CSE_ID = "62678cb02935948d8"

# Function to fetch Google search results
def google_search(query):
    try:
        url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={GOOGLE_API_KEY}&cx={GOOGLE_CSE_ID}"
        response = requests.get(url)
        
        if response.status_code != 200:
            return f"Error: Unable to fetch results (Status code: {response.status_code})"
        
        results = response.json()

        if 'error' in results:
            return f"Google API Error: {results['error'].get('message', 'Unknown error')}"

        snippets = [item.get('snippet', 'No snippet available') for item in results.get('items', [])]
        return "\n\n".join(snippets) if snippets else "No relevant information found on Google."

    except Exception as e:
        return f"An error occurred: {str(e)}"

def get_weather(city):
    st.write(f"Fetching weather for: {city}")  
    try:
        complete_api_link = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={weather_api_key}"
        
        api_link = requests.get(complete_api_link)
        api_data = api_link.json()
        
        if api_link.status_code == 200:
            temp_city = api_data['main']['temp'] - 273.15
            weather_desc = api_data['weather'][0]['description']
            humidity = api_data['main']['humidity']
            wind_speed = api_data['wind']['speed']
            date_time = datetime.now().strftime("%d %b %Y | %I:%M:%S %p")
            
            weather_response = f"Temperature: {temp_city:.2f}°C\n"
            weather_response += f"Weather Description: {weather_desc}\n"
            weather_response += f"Humidity: {humidity}%\n"
            weather_response += f"Wind Speed: {wind_speed} m/s\n"
            weather_response += f"Date & Time: {date_time}"
            
            st.write("Weather_bot:", weather_response)
        else:
            st.write(f"Error: {api_data.get('message', 'Unable to fetch weather data.')}")
    
    except requests.RequestException as e:
        st.write(f"Request error: {e}")
    except KeyError as e:
        st.write(f"Key error: {e}")
    except Exception as e:
        st.write(f"An unexpected error occurred: {e}")

# Function to generate response using Blenderbot model
def generate_chat_response(user_input):
    inputs = tokenizer.encode(user_input, return_tensors="pt", clean_up_tokenization_spaces=True)
    reply_ids = model.generate(inputs)
    bot_reply = tokenizer.decode(reply_ids[0], skip_special_tokens=True)
    st.write("Bot_reply:", bot_reply)

# Streamlit app interface
st.title("Multi-functional Chatbot")

# Create a sidebar for navigation
option = st.sidebar.selectbox(
    "Select the chatbot functionality",
    ["Student Q&A", "Weather Prediction", "Chit-chat"]
)

# Handle the selected option
if option == "Student Q&A":
    st.write("Ask a question related to your studies, and I will provide an answer based on Google!")
    user_question = st.text_input("Enter your question:")
    
    if user_question:
        with st.spinner("Finding the best answer..."):
            answer = google_search(user_question)
        st.write("**Answer:**", answer)

elif option == "Weather Prediction":
    st.write("Get the current weather for any city!")
    user_city = st.text_input("Enter city name:", "")
    
    if st.button("Get Weather") and user_city:
        st.write("Fetching weather data...")
        get_weather(user_city)
        
elif option == "Chit-chat":
    st.write("Have a casual conversation with the chatbot!")
    user_input = st.text_input("Enter your message:", "")
    
    if user_input:
        with st.spinner("Generating response..."):
            generate_chat_response(user_input)
