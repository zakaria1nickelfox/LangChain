#SEMENTIC-SEARCH

from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
load_dotenv()   

llm = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

input=["Zakaria is an excellent AI engineer",
       "Max has developed a great new product on web development",
       "Ron is a good data scientist",
       "Anna is a proficient software developer",
       "Lina is a skilled graphic designer"
       ]
input2 = "Rain Rain go away come again another day"
vector2 = llm.embed_query(input2)
vector = llm.embed_documents(input)

query = "Who will solve me AI problems"

vector_query = llm.embed_query(query)

similarity = cosine_similarity([vector_query], vector)[0]

index,score = sorted(list(enumerate(similarity)),key=lambda x:x[1])[-1]

print("Matched sentence from docs is ",input[index],"with score of",score)
