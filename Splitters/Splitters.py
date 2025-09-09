from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredMarkdownLoader  # Updated import

# Load the Markdown document
loader = UnstructuredMarkdownLoader("Splitters/markdown.md")  

loaded_docs = loader.load()

# Initialize the RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter.from_language(
    language="markdown",
    chunk_size=50,                      # Target chunk size (50 characters)
    chunk_overlap=10,                
)

# Split the document
chunks = text_splitter.split_documents(loaded_docs)

# Print chunks with index and length
print("Splitting Markdown Document into Chunks:",chunks)
