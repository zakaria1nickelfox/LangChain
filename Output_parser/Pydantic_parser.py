from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import Optional
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv

load_dotenv()

llm=HuggingFaceEndpoint(
    repo_id = "Qwen/Qwen2.5-7B-Instruct",
    task = "conversational",
)

model=ChatHuggingFace(llm=llm)

class Fictional_celebrity(BaseModel):
    name: str = Field(...,description="The name of the Fictional Celebrity ")
    age: int = Field(...,gt=15,description="The age of the Fictional Celebrity")
    superpower: str = Field(...,description="The superpower of the Fictional Celebrity")
    country: Optional[str] = Field(None, description="The country of the Fictional Celebrity")



parser= PydanticOutputParser(pydantic_object=Fictional_celebrity)

tempate= PromptTemplate(
    template="Generate a fictional Character with name, age, superpower etc from {Place} \n {format_instructions}",
    input_variables=["Place"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = tempate | model |parser

result=chain.invoke({"Place":"Russia"})

print(result.json())