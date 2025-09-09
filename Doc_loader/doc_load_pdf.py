from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import WebBaseLoader

pdf_loader=PyPDFLoader(file_path=r"Doc_loader\Datasets\data2.pdf")
web_loader = WebBaseLoader("https://en.wikipedia.org/wiki/Artificial_intelligence")


# docs=pdf_loader.load() #When pdf document is small in size
docs=list(pdf_loader.lazy_load()) #When pdf document is large in size
# 1. Initialize the loader with one or more URLs

# 2. Load the data
docs1 = web_loader.load()


print(docs1[0].page_content[:500])   # print first 500 chars


# print(docs[0].page_content)