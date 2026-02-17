from langchain_text_splitters import RecursiveCharacterTextSplitter
from loader import docs

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # chunk size (characters)
    chunk_overlap=200,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
)
all_splits = text_splitter.split_documents(docs)

if __name__ == "__main__":
    print(f"Split blog post into {len(all_splits)} sub-documents.")