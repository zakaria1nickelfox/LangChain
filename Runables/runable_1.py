from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv  
from pydantic import BaseModel, Field
from typing import List,Optional,Literal
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser,StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough, RunnableBranch, RunnableLambda,RunnableSequence



load_dotenv()   

llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    task="conversational",   
    temperature=0.7,)


model = ChatHuggingFace(llm=llm)

parser= StrOutputParser()

PromptTemplate_joke = PromptTemplate(
    template="Tell me a joke on {Topic}",
    input_variables=["Topic"]
    )



PromptTemplate_description = PromptTemplate(
    template="Give me a detailed description of {Topic}",
    input_variables=["Topic"]
)

joke = PromptTemplate_joke | model | parser

description = PromptTemplate_description | model | parser 
 
result = joke.invoke({"Topic":"Technology"})


description_result = description.invoke({"Topic":result})


def count_word(x):
    return len(x.split())

lmdbda=RunnableLambda(count_word)

parallel_chain = RunnableParallel({
    'description': PromptTemplate_description | model | parser,
    'words_count': RunnableLambda(lambda x: count_word(x))
    , 'jokes':RunnablePassthrough()
})


parallel_result = parallel_chain.invoke(result)


print(parallel_result)