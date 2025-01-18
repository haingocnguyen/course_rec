#chatbot_app.py



import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from typing import List, Dict
from models.gpt2_model import load_gpt2, generate_gpt2_response
from models.sentence_bert_model import load_sentence_bert, generate_embeddings
from models.other_model import load_other_model, generate_other_response
from indexing.faiss_index import create_faiss_index

from models.llama import load_llama, generate_llama_response

# Load the CSV into a pandas DataFrame
df = pd.read_csv('D:\\Thesis\\testwmodel\\data\\Online_Courses.csv')  # Replace with your actual CSV path
df.drop_duplicates(subset="Title", keep="first", inplace=True)
# Convert all non-string columns to strings to avoid type mismatch issues
for col in df.columns:
    if df[col].dtype != "object":  # Non-string columns
        df[col] = df[col].astype(str)

# Fill missing values with empty strings
df.fillna("", inplace=True)

# Process text data to combine Title, Short Intro, and Skills
def process_text_data(df):
    df['Title'].fillna('', inplace=True)
    df['Short Intro'].fillna('', inplace=True)
    df['Skills'].fillna('', inplace=True)

    df['combined_text'] = df['Title'] + " " + df['Short Intro'] + " " + df['Skills']
    return df['combined_text'].tolist(), df

# Prepare course data
combined_text, df = process_text_data(df)

# Function to search for similar courses
def search_courses(query, model, faiss_index, k=3):
    query_embedding = model.encode([query])
    distances, indices = faiss_index.search(query_embedding, k)

    recommended_courses = [
        {
            "Title": df.iloc[idx]["Title"],
            "URL": df.iloc[idx].get("URL", "No URL available"),
            "Short Intro": df.iloc[idx]["Short Intro"],
            "Category": df.iloc[idx].get("Category", "No category available"),
            "Instructors": df.iloc[idx].get("Instructors", "No instructors listed")
        }
        for idx in indices[0] if idx < len(df)
    ]

    return recommended_courses

# Main chatbot function to integrate course search, greeting, and normal response generation
def run_course_recommendation_chatbot(query: str, chat_history: List[Dict] = None, faiss_index=None, sentence_bert_model=None, model_name=None):
    if chat_history is None:
        chat_history = []

    # Append user query to chat history
    chat_history.append({'role': 'USER', 'message': query})

    # Load the specified language model dynamically
    if model_name == "gpt2":
        model, tokenizer = load_gpt2()
        generate_response_fn = generate_gpt2_response
    elif model_name == "llama":
        model, tokenizer = load_llama()
        generate_response_fn = generate_llama_response
    else:
        model, tokenizer = load_other_model()
        generate_response_fn = generate_other_response

    # First check if the query is related to course recommendations
    if any(keyword in query.lower() for keyword in ["provide details", "describe", "description", "summarize", "summary", "details", "cover", "what about"]):
        #llama_model, llama_tokenizer = load_llama()
        prompt = f"Provide details about the course: {query}"
        response = generate_llama_response(prompt, model, tokenizer)
        citations = None
    elif "machine learning" in query.lower():
        recommended_courses = search_courses("machine learning", sentence_bert_model, faiss_index)
        if recommended_courses:
            response = "Based on your inquiry about Machine Learning, I found some courses that might interest you:\n"
            citations = []
            for i, course in enumerate(recommended_courses):
                response += f"{i+1}. {course['Title']} - {course['Short Intro']} ({course['URL']})\n"
                citations.append({
                    'Title': course['Title'],
                    'URL': course['URL'],
                    'Short Intro': course['Short Intro']
                })
            response += "\nWould you like to explore one of these courses, or do you have any other questions?"
        else:
            response = "I couldn't find any machine learning courses matching your query. Could you be more specific or try another topic?"
            citations = None
    # Handle greetings or other general conversational queries
    elif "hello" in query.lower() or "hi" in query.lower():
        response = "Hi there! 😊 How can I assist you today? Feel free to ask about courses, or anything else you'd like to know!"
        citations = None
    elif "how are you" in query.lower():
        response = "I'm doing great, thanks for asking! How about you? 😊"
        citations = None
    elif "tell me about yourself" in query.lower():
        response = "I am a friendly chatbot here to help you find courses and answer any other questions you have. What can I help with today?"
        citations = None
    else:
        # Default case: search courses based on user query
        recommended_courses = search_courses(query, sentence_bert_model, faiss_index)

        if recommended_courses:
            response = "Based on your inquiry, I found some courses that might interest you:\n"
            citations = []
            for i, course in enumerate(recommended_courses):
                response += f"{i+1}. {course['Title']} - {course['Short Intro']} ({course['URL']})\n"
                citations.append({
                    'Title': course['Title'],
                    'URL': course['URL'],
                    'Short Intro': course['Short Intro']
                })
            response += "\nWould you like to explore one of these courses, or do you have any other questions?"
        else:
            response = "I'm sorry, I couldn't find any courses matching your query. Could you provide more details or try a different topic?"
            citations = None

    # Generate a response using the model
    final_response = generate_response_fn(response, model, tokenizer)

    # Append chatbot response to chat history
    chat_history.append({'role': 'CHATBOT', 'message': final_response})

    return chat_history, final_response

# Streamlit code for the chatbot app
def main():
    st.title("Course Recommendation Chatbot")
    st.subheader("Ask me anything about courses!")

    # Initialize session state for storing the chat history and model
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Load models and create FAISS index on the first run
    if "faiss_index" not in st.session_state:
        sentence_bert_model = load_sentence_bert()
        faiss_index, _ = create_faiss_index(sentence_bert_model, combined_text)
        st.session_state.faiss_index = faiss_index
        st.session_state.sentence_bert_model = sentence_bert_model
    if "llama_model" not in st.session_state or "llama_tokenizer" not in st.session_state:
        llama_model, llama_tokenizer = load_llama()  # Load only once
        st.session_state.llama_model = llama_model
        st.session_state.llama_tokenizer = llama_tokenizer
    

    # Display previous chat history
    for message in st.session_state.chat_history:
        if message['role'] == 'USER':
            st.write(f"**You**: {message['message']}")
        else:
            st.write(f"**Bot**: {message['message']}")

    # Input box for the user query
    user_input = st.text_input("You:", "")

    if user_input:
        # Get the chatbot response
        st.session_state.chat_history, chatbot_response = run_course_recommendation_chatbot(
            user_input, 
            st.session_state.chat_history, 
            faiss_index=st.session_state.faiss_index, 
            sentence_bert_model=st.session_state.sentence_bert_model, 
            model_name="llama"
        )

        # Display chatbot response
        st.write(f"**Bot**: {chatbot_response}")

if __name__ == "__main__":
    main()
