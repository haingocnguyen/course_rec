from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import streamlit as st

# Initialize LLaMA model and tokenizer
@st.cache_resource
def load_llama():
    """
    Load the LLaMA 3.2 1B model and tokenizer with a Hugging Face token.
    """
    model_name = "meta-llama/Llama-3.2-1B-Instruct"  
    hf_token = "hf_xx"  

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token)

    return model, tokenizer

# Function to generate LLaMA response
def generate_llama_response(prompt: str, model, tokenizer, temperature=0.5):
    """
    Generate a response from the LLaMA 3.2 1B model based on the prompt.
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


