from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

loader=TextLoader("Doc_loader/Datasets/data.txt", encoding="utf8")

text=loader.load()[0].page_content

recursive_text_splitter = RecursiveCharacterTextSplitter(
    separators=[""],  # Empty separator for pure character-based splitting
    chunk_size=10,
    chunk_overlap=2,
    length_function=len
)

chunks_1 = recursive_text_splitter.split_text(text)

text_splitter = CharacterTextSplitter(
    separator=" ",
    chunk_size=10,
    chunk_overlap=2,
)

chunks = text_splitter.split_text(text)

print(f"Number of chunks: {len(chunks)}")
print(f"Number of chunks_recursive: {len(chunks_1)}")