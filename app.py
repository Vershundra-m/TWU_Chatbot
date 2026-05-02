import streamlit as st
import os
import sys

sys.path.append('.')
from scraper import get_clean_text, chunk_text
from database import VectorStore
import google.generativeai as genai
import config

# Page configuration
st.set_page_config(
    page_title="Informatics AI Assistant",
    page_icon="🎓",
    layout="wide"
)

# CCS for Webpage 
st.markdown("""
<style>
    .stButton > button {
        background-color: #850928;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
    }
    .stButton > button:hover {
        background-color: #6B0720;
        color: white;
    }
    .user-message {
        background-color: #F5F5F5;
        padding: 12px;
        border-radius: 12px;
        margin: 8px 0;
        border-left: 4px solid #850928;
    }
    .assistant-message {
        background-color: #FAF5F6;
        padding: 12px;
        border-radius: 12px;
        margin: 8px 0;
        border-left: 4px solid #C4A23F;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'question_count' not in st.session_state:
    st.session_state.question_count = 0
if 'db' not in st.session_state:
    with st.spinner("Loading Informatics AI Assistant..."):
        clean_text = get_clean_text()
        chunks = chunk_text(clean_text)
        db = VectorStore()
        db.connect_index()
        db.upload_chunks(chunks)
        st.session_state.db = db

# Setup Gemini API
genai.configure(api_key=config.GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash-lite')

# Header
col1, col2 = st.columns([3, 1])
with col1:
    st.title("Informatics AI Assistant")
with col2:
    st.markdown("[Help](https://twu.edu/technology/) | [Info]")

st.markdown("---")

# Search bar
user_input = st.text_input("Ask a question", placeholder="Type your question here...", label_visibility="collapsed")
ask_button = st.button("Ask", use_container_width=True)

# FAQ buttons
st.write("Frequently Asked Questions:")
faq_cols = st.columns(5)
faqs = [
    ("Deadlines", "What are the application deadlines?"),
    ("GPA Reqs", "What are the GPA requirements?"),
    ("Emphases", "What emphases are offered?"),
    ("Credits", "How many credits are required?"),
    ("Degree", "What degree is offered?")
]
for idx, (label, question) in enumerate(faqs):
    with faq_cols[idx]:
        if st.button(label):
            user_input = question

# Chat history 
st.write("Chat History")
chat_container = st.container()
with chat_container:
 for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Clear button
col_clear, _ = st.columns([1, 5])
with col_clear:
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.question_count = 0
        st.rerun()

# Sidebar
with st.sidebar:
    st.write(f"**Session Progress**")
    st.progress(st.session_state.question_count / 10)
    st.caption(f"Questions used: {st.session_state.question_count}/10")
    
    st.markdown("---")
    st.write("**Assistant Info**")
    st.caption("Model: Gemini 2.5 Flash-Lite")
    st.caption("Vector Search: Pinecone")
    
    st.markdown("---")
    if st.button("Reset Session", use_container_width=True):
        st.session_state.messages = []
        st.session_state.question_count = 0
        st.rerun()

# Footer
st.markdown("---")
st.caption("For accommodations, contact the TWU Service Desk.")

# Generate response function
def generate_answer(question):
    retrieved = st.session_state.db.query(question, top_k=3)
    
    if not retrieved:
        return "TWU is updating information at this time. Please check back shortly."
    
    context = "\n\n---\n\n".join([chunk['text'] for chunk in retrieved])
    prompt = f"""Answer the question based on the context below. If the answer is not clearly stated, say "TWU is updating information at this time. Please check back shortly."

CONTEXT: {context}
QUESTION: {question}
ANSWER:"""
    
    response = model.generate_content(prompt)
    return response.text

# Processing
if (user_input and user_input.strip()) or ask_button:
    if st.session_state.question_count >= 10:
        st.warning("You have reached the 10-question limit. Please reset the session.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.spinner("Assistant is thinking..."):
            answer = generate_answer(user_input)
        
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.session_state.question_count += 1
        st.rerun()