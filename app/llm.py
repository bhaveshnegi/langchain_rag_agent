import os
from dotenv import load_dotenv
from typing import Any, List, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from cache import get_llm_cache, set_llm_cache, get_hash

# Load environment variables from .env file
load_dotenv()

llm_provider = os.getenv("LLM", "HF").upper()

if llm_provider == "AWS":
    from langchain_aws import ChatBedrock
    
    # Initialize AWS Bedrock model
    model_base = ChatBedrock(
        model_id="mistral.mistral-7b-instruct-v0:2",
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
    model_base = ChatHuggingFace(llm=llm)
    print(f"--- LLM initialized with Hugging Face (Mistral-7B) ---")

class CachedChatModel(BaseChatModel):
    model_to_wrap: BaseChatModel
    
    def __init__(self, model_to_wrap: BaseChatModel, **kwargs):
        super().__init__(model_to_wrap=model_to_wrap, **kwargs)

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs: Any) -> ChatResult:
        # Convert messages to a single string to hash
        prompt_str = "".join([f"{m.type}:{m.content}" for m in messages])
        cached_res = get_llm_cache(prompt_str)
        
        if cached_res:
            print(f"--- LLM (Prompt) Cache HIT ---")
            # Reconstruct the expected response format
            from langchain_core.messages import AIMessage
            from langchain_core.outputs import ChatGeneration
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=cached_res))])

        print(f"--- LLM (Prompt) Cache MISS ---")
        result = self.model_to_wrap._generate(messages, stop, **kwargs)
        
        # Cache the result content
        if result.generations:
            answer = result.generations[0].message.content
            set_llm_cache(prompt_str, answer)
            
        return result

    @property
    def _llm_type(self) -> str:
        return "cached_model"

# Initialize the cached model
if llm_provider == "AWS":
    model = CachedChatModel(model_base)
else:
    model = CachedChatModel(model_base)

if __name__ == "__main__":
    # Quick test to see if it's working
    print(f"Provider: {llm_provider}")