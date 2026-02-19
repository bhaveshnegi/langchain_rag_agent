# import bs4
# from langchain_community.document_loaders import WebBaseLoader

# # Only keep post title, headers, and content from the full HTML.
# bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
# # loader = WebBaseLoader(
# #     web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
# #     bs_kwargs={"parse_only": bs4_strainer},
# # )
# docs = loader.load()

# if __name__ == "__main__":
#     assert len(docs) == 1
#     print(f"Total characters: {len(docs[0].page_content)}")

import os
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Get absolute path to the data folder
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(CURRENT_DIR), "data")

loader = PyPDFDirectoryLoader(DATA_DIR)
docs = loader.load()

if __name__ == "__main__":
    print(f"Pages loaded: {len(docs)}")
    print(f"Total characters: {sum(len(doc.page_content) for doc in docs)}")
