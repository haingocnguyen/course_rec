from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import PeftModel
import streamlit as st

# Load fine-tuned LLaMA model and tokenizer
@st.cache_resource
def load_llama():
    # Base LLaMA model path and fine-tuned adapter path
    base_model_path = "meta-llama/Llama-3.2-1B-Instruct"  # Replace with the base model path
    adapter_path = "D:\\Thesis\\testwmodel\\llama3.2instruct"
    hf_token = "hf_pSlGmFdVVWThvNwXVuYluSnJEAubrhEkV"

    # Load the tokenizer
    tokenizer = LlamaTokenizer.from_pretrained(base_model_path, use_auth_token=hf_token)

    # Load the base model and apply the adapter
    base_model = LlamaForCausalLM.from_pretrained(base_model_path, use_auth_token=hf_token)
    model = PeftModel.from_pretrained(base_model, adapter_path)

    return model, tokenizer

# Function to generate response using the LLaMA model
def generate_llama_response(prompt: str, model, tokenizer, temperature=0.5):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs["input_ids"],
        max_new_tokens=150,
        temperature=temperature,
        no_repeat_ngram_size=2,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response
