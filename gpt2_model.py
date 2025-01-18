#models/gpt2_model.py

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import streamlit as st

# Initialize GPT2 model and tokenizer
@st.cache_resource
def load_gpt2():
    tokenizer = GPT2Tokenizer.from_pretrained("D:\\Thesis\\testwmodel\\gpt2-final-model")
    model = GPT2LMHeadModel.from_pretrained("D:\\Thesis\\testwmodel\\gpt2-final-model")
    return model, tokenizer

# Function to generate GPT2 response
def generate_gpt2_response(prompt: str, model, tokenizer, temperature=0.5):
    """
    Generate a response from the GPT-2 model based on the prompt.
    """
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs,
        max_new_tokens=100,
        temperature=temperature,
        no_repeat_ngram_size=2,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response
