import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

llm_provider = os.getenv("LLM", "HF").upper()

if llm_provider == "AWS":
    from langchain_aws import ChatBedrock
    
    # Initialize AWS Bedrock model
    model = ChatBedrock(
        model_id="qwen.qwen3-235b-a22b-2507-v1:0",
        region_name="ap-south-1",
    )
    print("--- LLM initialized with AWS Bedrock ---")

else:
    # Default to Hugging Face
    from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
    
    hf_token = os.getenv("HUGGINGFACE_API_KEY")
    if hf_token:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token
        os.environ["HF_TOKEN"] = hf_token
        os.environ["HUGGINGFACE_API_TOKEN"] = hf_token

    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        huggingfacehub_api_token=hf_token,
        temperature=0.1,
        max_new_tokens=1024,
    )
    model = ChatHuggingFace(llm=llm)
    print(f"--- LLM initialized with Hugging Face (Mistral-7B) ---")

if __name__ == "__main__":
    # Quick test to see if it's working
    print(f"Provider: {llm_provider}")