import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

load_dotenv()

endpoint_llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="conversational",   
    temperature=0.7,
)

chat = ChatHuggingFace(llm=endpoint_llm)


msgs = [
    ("system", "You are a helpful assistant."),
    ("human", "What is the most visited country by tourists?")
]
print(chat.invoke(msgs).content)
