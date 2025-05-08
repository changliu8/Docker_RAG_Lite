from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_ollama.llms import OllamaLLM
import sys
import requests
import os
import time
import json


embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
faiss_path = "faiss_index"

vector_db = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)

def retrive_context(query, k=3, score_threshold=0.8):
    retrieved_context = vector_db.similarity_search(
        query,
        k = k,
        score_threshold = score_threshold
    )
    return retrieved_context


OLLAMA_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

def answer_question(question, context, llm_api,data):
    # Reformat context
    formatted_context = "\n".join([doc.page_content for doc in context])

    # Prompt Template
    prompt = f"""
    You are an expert research assistant specializing in answering questions about research papers.

    Task: Answer the question based on the provided context, with detail explaination and reasoning.

    Instructions:
    * Be concise and accurate.
    * If the context does not contain the answer, say EXACTLY "I cannot answer confidently"
    * If the question is unrelated to the context, say EXACTLY "NA"
    * If the question asks for a yes/no answer, provide it and explain your reasoning shortly.

    Context:
    {formatted_context}

    Question:
    {question}

    Answer:
    """

    # Generate answer using the LLM
    try:
        response = requests.post(f"{llm_api}/api/generate", json=data)
        data = response.json()
        return data.get("response", "No response from model")
    except Exception as e: 
        print(f"Error during LLM call: {e}")
        return "Error processing the request."

def response(query, data, k=10, score_threshold=0.8):
    retrieved_context = retrive_context(query, k=k, score_threshold=score_threshold)
    if not retrieved_context:
        return "No relevant context found."
    
    # Answer the question using the LLM
    response = answer_question(query, retrieved_context, OLLAMA_URL,data)
    return response

def wait_for_llama_model(base_url, retries=25, delay=20):
    for i in range(retries):
        try:
            res = requests.get(f"{base_url}/api/tags")
            if res.ok and 'llama3' in res.text:
                print("Model is ready.")
                return
        except requests.RequestException:
            pass
        print(f"Waiting for model to be available... attempt {i+1}")
        time.sleep(delay)
    raise TimeoutError("Timed out waiting for the model.")

if __name__ == "__main__":
    wait_for_model()
    model = "llama3"
    query = sys.argv[1]
    data = {"model":model, "prompt":query, "stream":False}
    print("Your Question is : ", query)
    answer = response(query, data, k=10, score_threshold=0.8)
    print("The answer is : ", answer)