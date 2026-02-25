from langchain_aws import ChatBedrock

llm = ChatBedrock(
    model_id="qwen.qwen3-235b-a22b-2507-v1:0",
    region_name="ap-south-1",
)

response = llm.invoke("Explain what RAG is in one short sentence.")
print(response.content)