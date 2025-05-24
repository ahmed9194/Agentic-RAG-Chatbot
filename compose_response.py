from openai import OpenAI
import os
os.environ["OPENAI_API_KEY"] = "your-key-here"

def generate_response(query, retrieved_chunks):
    context = "\n".join(retrieved_chunks)
    prompt = f"""You are a helpful travel assistant. Based on the information below, answer the user's query in a friendly tone.

### Query:
{query}

### Context:
{context}

### Answer:"""

    import openai
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["choices"][0]["message"]["content"]