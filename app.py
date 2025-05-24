# app.py - Fully self-contained Streamlit app for Agentic RAG Travel Chatbot using OpenAI embeddings

import os
import streamlit as st
import pandas as pd
import re
import json
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
import openai
from tqdm import tqdm

# Set your OpenAI API key here
openai.api_key = "YOUR_OPENAI_API_KEY"

# === Utility Functions ===
def get_embedding(text):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response['data'][0]['embedding']

@st.cache_resource
def load_qdrant_and_index():
    # Load and preprocess data
    df = pd.read_csv(r"C:\Users\user\Desktop\tripadvisor_hotel_reviews.csv")
    df = df.dropna()

    def clean_text(text):
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'[^a-zA-Z0-9\s.,!?]', '', text)
        return text.strip()

    df['clean_review'] = df['Review'].apply(clean_text)

    chunks = []
    chunk_id = 0
    for i, row in tqdm(df.iterrows(), total=len(df)):
        sentences = re.split(r'(?<=[.!?]) +', row['clean_review'])
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk.split()) + len(sentence.split()) < 500:
                current_chunk += " " + sentence
            else:
                chunks.append({"id": chunk_id, "text": current_chunk.strip()})
                chunk_id += 1
                current_chunk = sentence
        if current_chunk:
            chunks.append({"id": chunk_id, "text": current_chunk.strip()})
            chunk_id += 1

    # Index in Qdrant (in-memory)
    qdrant = QdrantClient("localhost", port=6333)
    try:
        qdrant.recreate_collection(
            collection_name="travel_chunks",
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
        )
    except:
        pass

    vectors = [get_embedding(chunk["text"]) for chunk in chunks]
    payloads = [{"text": chunk["text"]} for chunk in chunks]
    ids = [chunk["id"] for chunk in chunks]

    qdrant.upsert(collection_name="travel_chunks", points=[{
        "id": id,
        "vector": vec,
        "payload": payload
    } for id, vec, payload in zip(ids, vectors, payloads)])

    return qdrant

def retrieve_chunks(query, qdrant, top_k=5):
    vector = get_embedding(query)
    hits = qdrant.search(collection_name="travel_chunks", query_vector=vector, limit=top_k)
    return [hit.payload["text"] for hit in hits]

def generate_response(query, retrieved_chunks):
    context = "\n".join(retrieved_chunks)
    prompt = f"""You are a helpful travel assistant. Based on the information below, answer the user's query in a friendly and informative tone.

### Query:
{query}

### Context:
{context}

### Answer:
"""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# === Streamlit UI ===
st.set_page_config(page_title="🌍 Travel Guide Chatbot", layout="centered")
st.title("🌍 Travel Guide Chatbot")
query = st.text_input("Ask a travel-related question:")

if query:
    with st.spinner("Processing your request..."):
        qdrant = load_qdrant_and_index()
        retrieved = retrieve_chunks(query, qdrant)
        response = generate_response(query, retrieved)
    st.markdown("---")
    st.subheader("✈️ Response:")
    st.success(response)
