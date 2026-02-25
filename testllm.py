from langchain_aws import ChatBedrock

llm = ChatBedrock(
    model_id="mistral.mistral-7b-instruct-v0:2",
    region_name="ap-south-1",
)

response = llm.invoke("Explain what RAG is in one short sentence.")
print(response.content)