#models/sentence_bert_model.py
from sentence_transformers import SentenceTransformer
import streamlit as st

# Initialize Sentence-BERT model and generate embeddings
@st.cache_resource
def load_sentence_bert():
    return SentenceTransformer('paraphrase-MiniLM-L6-v2')

def generate_embeddings(model, text_data):
    """Generate embeddings using Sentence-BERT."""
    return model.encode(text_data)
