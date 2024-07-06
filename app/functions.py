import os
from dotenv import load_dotenv

api_keys = "../.env"
load_dotenv(api_keys)
pinecone_api_key = os.environ.get("PINECONE_API_KEY")

from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import openai
import streamlit as st

model = SentenceTransformer("all-MiniLM-L6-v2")

pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index("ai-assistant")


def find_match(input):
    input_em = model.encode(input).tolist()
    result = index.query(vector=input_em, top_k=2, includeMetadata=True)
    return (
        result["matches"][0]["metadata"]["text"]
        + "\n"
        + result["matches"][1]["metadata"]["text"]
    )


def query_refiner(conversation, query):
    response = openai.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:",
        temperature=0.7,
        max_tokens=256,
        top_p=1,  # https://community.openai.com/t/cheat-sheet-mastering-temperature-and-top-p-in-chatgpt-api/172683
        frequency_penalty=0,
        presence_penalty=0,
    )
    return response.choices[0].text.strip()


def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state["responses"]) - 1):
        conversation_string += "Human: " + st.session_state["requests"][i] + "\n"
        conversation_string += "Bot: " + st.session_state["responses"][i + 1] + "\n"
    return conversation_string
