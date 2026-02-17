import os
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

# Load environment variables from .env file
load_dotenv()

# Use the API key from environment variables
hf_token = os.getenv("HUGGINGFACE_API_KEY")
if hf_token:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token
    os.environ["HF_TOKEN"] = hf_token
    os.environ["HUGGINGFACE_API_TOKEN"] = hf_token

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    temperature=0.7,
    max_new_tokens=1024,
)
model = ChatHuggingFace(llm=llm)

if __name__ == "__main__":
    # Quick test to see if it's working
    print(f"Model initialized with token: {hf_token[:5]}..." if hf_token else "No token found")