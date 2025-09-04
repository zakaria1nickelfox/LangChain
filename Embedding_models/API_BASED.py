import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()


emb = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004"
)

q_vec = emb.embed_query( "What is the capital of India?",output_dimensionality=32 )
print("first five vector", q_vec[:5])
print("lenght of vector", len(q_vec))

# For documents (list of strings):
docs = ["Delhi is the capital of India.", "Paris is the capital of France."]
doc_vecs = emb.embed_documents(docs)

print("lenght of vectors (Default size of output by models/text-embedding-004 )", len(doc_vecs[0]))
