import streamlit as st
import torch
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import util
from together import Together
from fpdf import FPDF  # For generating PDFs
import os

# Download required NLTK data files
import streamlit as st
import nltk

# Download required NLTK data files
# Download required NLTK data files
st.write("Downloading NLTK data files...")
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')  # Add this for Open Multilingual WordNet
st.write("NLTK data files downloaded successfully!")
# Configuration
CODES = {
    "Tunisia": "TN.txt",
    "United States": "US.txt",
    "United Kingdom": "UK.txt",
    "Canada": "CA.txt",
    "Australia": "AU.txt"
}
MODEL_NAME = "dangvantuan/sentence-camembert-large"
LLM_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo-128K"
TOGETHER_API_KEY = st.secrets["API_KEY"]

# Load Models
@st.cache_resource
def load_similarity_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    model = AutoModel.from_pretrained(MODEL_NAME)
    return tokenizer, model

@st.cache_resource
def get_llm_client():
    return Together(api_key=TOGETHER_API_KEY)

@st.cache_resource
def load_chatbot():
    lemmatizer = WordNetLemmatizer()
    intents = json.load(open('intents.json', encoding='utf-8'))
    words = pickle.load(open('words.pkl', 'rb'))
    classes = pickle.load(open('classes.pkl', 'rb'))
    model = load_model('chatbot_model10.h5')
    return lemmatizer, intents, words, classes, model

# Utility Functions
def read_file(filename):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        st.error(f"File {filename} not found!")
        return None

def encode_text(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

def generate_report(code1, code2, client):
    prompt = f"""
    Analyse comparative des codes d'éthique d'ingénieurs:
    
    Texte 1:
    {code1}
    
    Texte 2:
    {code2}
    
    Génère un rapport structuré en français avec:
    1. Points communs principaux (5 points maximum)
    2. Différences clés (5 points maximum)
    3. Recommandations d'amélioration pour chaque code
    4. Perspectives globales
    """
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "system", "content": "Expert en analyse comparative de codes éthiques professionnels."},
                  {"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def chatbot_response(user_input, lemmatizer, intents, words, classes, model):
    sentence_words = nltk.word_tokenize(user_input)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    bag = [1 if word in sentence_words else 0 for word in words]
    res = model.predict(np.array([bag]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    if not results:
        return "Je ne suis pas sûr de comprendre. Pouvez-vous reformuler ?"
    tag = classes[results[0][0]]
    for intent in intents['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return "Je ne peux pas répondre à cette question."

def create_pdf(report, filename):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Properly handle UTF-8 encoding
    report = report.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 10, report)
    pdf.output(filename)
# Streamlit UI
st.set_page_config(page_title="AI Ethics & Chatbot", layout="wide")

# Custom CSS for background and UI enhancements with blue palette and animations
st.markdown(
    """
    <style>
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes slideInLeft {
        from { transform: translateX(-30px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes slideInRight {
        from { transform: translateX(30px); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .stApp {
        background-image:linear-gradient(rgba(0,0,0,0.6),rgba(0,0,0,0.6)), url("https://wallpapers.com/images/hd/law-background-ycjay4xmeroqk3bm.jpg");
        background-size: cover;
        background-position: center;
    }
    
    .stTextInput>div>div>input {
        background-color: #ffffff;
        color: #333;
        border-radius: 20px;
        padding: 10px 15px;
        border: 1px solid #ddd;
        transition: all 0.3s ease;
    }
    
    .stTextInput>div>div>input:focus {
        box-shadow: 0 0 8px rgba(0, 120, 212, 0.6);
        border: 1px solid #0078D4;
    }
    
    .stButton>button {
        background-color: #0078D4;
        color: white;
        border-radius: 20px;
        padding: 10px 20px;
        font-size: 16px;
        border: none;
        transition: all 0.3s ease;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
        animation: pulse 2s infinite;
    }
    
    .stButton>button:hover {
        background-color: #106EBE;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
    }
    
    # In the custom CSS section
.chat-container {
    display: flex;
    flex-direction: column;
    gap: 10px;
    height: 60vh;  # This creates a fixed height that might be showing as blank
    padding: 20px;
    background-color: rgba(255, 255, 255, 0.9);  # This creates the white background
    border-radius: 15px;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    overflow-y: auto;
    margin-bottom: 10px;
}
    
    .chat-bubble {
        max-width: 70%;
        padding: 12px 16px;
        border-radius: 20px;
        margin-bottom: 8px;
        word-wrap: break-word;
        opacity: 0;
        animation: fadeIn 0.5s ease forwards;
        animation-delay: 0.1s;
    }
    
    .user-bubble {
        background: linear-gradient(135deg, #0078D4 0%, #106EBE 100%);
        color: white;
        align-self: flex-end;
        margin-left: auto;
        border-bottom-right-radius: 4px;
        animation: slideInRight 0.5s ease forwards;
    }
    
    .bot-bubble {
        background: linear-gradient(135deg, #f0f0f0 0%, #e0e0e0 100%);
        color: #333;
        align-self: flex-start;
        margin-right: auto;
        border-bottom-left-radius: 4px;
        animation: slideInLeft 0.5s ease forwards;
    }
    
    .input-container {
        display: flex;
        background-color: white;
        padding: 10px;
        border-radius: 15px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        margin-top: 10px;
        animation: fadeIn 0.8s ease;
    }
    
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #0078D4;
        animation: fadeIn 1s ease;
    }
    
    /* Hide default placeholder and label */
    div[data-testid="stForm"] {
        border: none;
        padding: 0;
    }
    
    .element-container:has(div[data-testid="stVerticalBlock"]) input {
        opacity: 1;
    }
    
    .stTextInput > div > div > input::placeholder {
        opacity: 1;
    }
    
    .css-9ycgxx, .css-r421ms {
        display: none;
    }
    
    /* Make the input look like a messenger input */
    div.stTextInput > div:first-child {
        width: 100%;
    }
    
    .send-container {
        display: flex;
        align-items: center;
    }
    
    .empty-bubble {
        text-align: center;
        color: #888;
        margin-top: 30vh;
        transform: translateY(-50%);
        animation: pulse 3s infinite;
    }
    
    /* Progress bar animation */
    .stProgress > div > div {
        background: linear-gradient(90deg, #0078D4, #10A5F5, #0078D4);
        background-size: 200% 200%;
        animation: gradientShift 3s ease infinite;
    }
    
    /* Metric value animation */
    [data-testid="metric-value"] {
        animation: fadeIn 1s ease;
    }
    
    /* Selectbox styling */
    .stSelectbox [data-baseweb="select"] {
        border-radius: 10px;
        transition: all 0.3s ease;
    }
    
    .stSelectbox [data-baseweb="select"]:hover {
        border-color: #0078D4;
        box-shadow: 0 0 5px rgba(0, 120, 212, 0.3);
    }
    
    /* Sidebar animation */
    .css-1d391kg, .css-1lcbmhc {
        animation: fadeIn 0.8s ease;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choisissez une fonctionnalité", ["⚖️ Analyseur de code d'éthique", "🤖 ETHOBOT"])

if page == "⚖️ Analyseur de code d'éthique":
    st.title("⚖️ Analyseur mondial d’éthique de l’ingénierie")
    tokenizer, model = load_similarity_model()
    llm_client = get_llm_client()
    
    col1, col2 = st.columns(2)
    with col1:
        code1 = st.selectbox("Sélectionnez le premier code", options=list(CODES.keys()), key="code1")
    with col2:
        code2 = st.selectbox("Sélectionnez le deuxième code", options=list(CODES.keys()), index=1, key="code2")
    
    if st.button("🔢 Calculer la similarité"):
        with st.spinner("Calcul de similarité..."):
            text1, text2 = read_file(CODES[code1]), read_file(CODES[code2])
            if text1 and text2:
                emb1, emb2 = encode_text(text1, tokenizer, model), encode_text(text2, tokenizer, model)
                similarity = util.cos_sim(torch.tensor(emb1), torch.tensor(emb2)).item()
                st.metric("Score de similarité", f"{similarity:.0%}")
                st.progress(similarity)
    
    if st.button("📊 Générer un rapport"):
        with st.spinner("Générer une analyse basée sur l'IA..."):
            text1, text2 = read_file(CODES[code1]), read_file(CODES[code2])
            if text1 and text2:
                report = generate_report(text1, text2, llm_client)
                st.subheader("Rapport d'analyse comparative")
                st.markdown(report)
                
                # Save report as PDF
                pdf_filename = f"ethics_report_{code1}_vs_{code2}.pdf"
                create_pdf(report, pdf_filename)
                with open(pdf_filename, "rb") as f:
                    st.download_button("📥 Télécharger le rapport", data=f, file_name=pdf_filename, mime="application/pdf")

elif page == "🤖 ETHOBOT":
    st.title("🤖 ETHOBOT")
    lemmatizer, intents, words, classes, model = load_chatbot()
    
    # Initialize session state for chat history
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
    
    # Function to handle sending messages
    def send_message():
        if st.session_state.chat_input:
            user_input = st.session_state.chat_input
            response = chatbot_response(user_input, lemmatizer, intents, words, classes, model)
            st.session_state['chat_history'].append(("user", user_input))
            st.session_state['chat_history'].append(("bot", response))
            st.session_state.chat_input = ""  # Clear the input box
    
    # Create chat container
    chat_container = st.container()
    
    # Display chat history
    with chat_container:
        st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
        if not st.session_state['chat_history']:
            st.markdown("<div class='empty-bubble'>Envoyez un message pour commencer la conversation</div>", unsafe_allow_html=True)
        else:
            for i, (role, message) in enumerate(st.session_state['chat_history']):
                animation_delay = f"animation-delay: {i * 0.2}s;"
                if role == "user":
                    st.markdown(f"<div class='chat-bubble user-bubble' style='{animation_delay}'>{message}</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='chat-bubble bot-bubble' style='{animation_delay}'>{message}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Create a simple input container at the bottom
    with st.container():
        col1, col2 = st.columns([5, 1])
        
        with col1:
            st.text_input("", key="chat_input", placeholder="Écrivez votre message...", on_change=send_message, label_visibility="collapsed")
        
        with col2:
            if st.button("Envoyer", on_click=send_message):
                pass  # The actual sending happens in the on_change/on_click functions