from langchain_community.document_loaders import TextLoader,CSVLoader
from langchain_huggingface import ChatHuggingFace , HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser   
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFaceEndpoint(repo_id="Qwen/Qwen2.5-7B-Instruct", task="text-generation", temperature=0.7,)

model=ChatHuggingFace(llm=llm)

parser=StrOutputParser()

Prompt_temp=PromptTemplate(
    template="Give me 4 pointer detailed description of {Topic}",
    input_variables=["Topic"]
)

chain=Prompt_temp | model | parser
result=chain.invoke({"Topic":"Technology"})

with open(r"Doc_loader\Datasets\data.txt","w",encoding="utf8") as f:
    f.write(result)




csv_loader=CSVLoader(file_path=r"Doc_loader\Datasets\sample_data.csv",encoding="utf8")
loader=TextLoader(file_path=r"Doc_loader\Datasets\data.txt",encoding="utf8")

docs=loader.load()
csv_docs=csv_loader.load()


print(docs)
print(csv_docs)