from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv  
from pydantic import BaseModel, Field
from typing import List,Optional,Literal
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser,StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough, RunnableBranch, RunnableLambda,RunnableSequence
import json

load_dotenv()


class ReviewSentiment(BaseModel):
    sentiment: Literal['positive', 'negative', 'neutral'] = Field(..., description="The sentiment of the review")

parser_pydantic=PydanticOutputParser(pydantic_object=ReviewSentiment)


llm= HuggingFaceEndpoint(
    repo_id = "Qwen/Qwen2.5-7B-Instruct",   
    task = "conversational",
    temperature=0.7,)

model=ChatHuggingFace(llm=llm)

review="""
The movie was decent The plot was engaging, 
the characters were not well-developed, and the cinemat ography was decents
.But I hate the ending and the music was too loud at times.
 Overall, it was a awfull experience experience .
"""

prompt_review = PromptTemplate(
    template="provide a sentiment value of  (positive, negative, neutral) of  Review: {review}  \n {template_instructions}",
    input_variables=["review"],
    partial_variables={"template_instructions":parser_pydantic.get_format_instructions()}
)

prompt_success = PromptTemplate(
    template="The review is {sentiment}. Write a response to thank the reviewer for their {sentiment} feedback.",
    variables=["sentiment"]
)

prompt_failure = PromptTemplate(
    template="The review is {sentiment}. Write a response to thank the reviewer for their {sentiment} feedback.",
    variables=["sentiment"]
)

str_parser=StrOutputParser()

model_chain=RunnableSequence( prompt_review , model , parser_pydantic)


Branchings=RunnableBranch(
    (lambda x:x.sentiment=="positive" , RunnableSequence(prompt_success , model ,str_parser)),
    (lambda x:x.sentiment=="negative" , RunnableSequence(prompt_failure , model ,str_parser)),
    RunnableLambda(lambda _:"Confuse Dumb Ass User get your mind right")
)

final_chain= model_chain | Branchings

print(final_chain.invoke({"review":review}))