from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv 
from pydantic import BaseModel, Field
from typing import List, Optional
import json

load_dotenv()   

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3,max_output_tokens=100)

result = model.invoke("Give me the review of dhoom 2 bollywood movie") # Load environment variables from .env file

class Review(BaseModel):
    tiitle: str = Field(..., description="Title of the movie")
    review: str = Field(..., description="Review of the movie")
    rating: Optional[str] = Field(None, description="Rating out Good or Bad or Average")

model_structured =model.with_structured_output(Review)

result_structured = model_structured.invoke(result.content).__dict__


print(result_structured)