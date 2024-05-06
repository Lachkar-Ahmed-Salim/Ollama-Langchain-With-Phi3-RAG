from langchain_community.embeddings import OllamaEmbeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")
text = "This is a test document."
query_result = embeddings.embed_query(text)
msg=query_result[:5]
print(msg)