#models/others_model.py

from transformers import AutoModelForCausalLM, AutoTokenizer

def load_other_model():
    """Load another model (e.g., GPT-Neo) and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B")
    return model, tokenizer

def generate_other_response(query: str, model, tokenizer):
    """Generate a response from GPT-Neo (or any other model)."""
    inputs = tokenizer.encode(query, return_tensors="pt")
    outputs = model.generate(
        inputs,
        max_length=100,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response
