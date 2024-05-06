from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load and process the PDF
loader = PyPDFLoader("GUIDE D’ASSURANCE DE LA PME.pdf")
documents = loader.load()
print("Le fichier a été uploadé")
text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
texts = text_splitter.split_documents(documents)
print("Le fichier divisé en plusieurs sous documents")

# Generate embeddings and initialize a retriever
embeddings = OllamaEmbeddings(model="mxbai-embed-large", show_progress=True)
db = FAISS.from_documents(texts, embeddings)
retriever = db.as_retriever()

# Create a prompt template
template = """Répondez à la question uniquement sur la base du contexte suivant:

    {context}

    Question: {question}
    """
prompt = ChatPromptTemplate.from_template(template)

# Define the LLM and the full chain
model = ChatOllama(model="phi3", num_predict=2000)

def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | prompt
    | model
    | StrOutputParser()
)


question = input("Votre question : ")
response=chain.invoke({"question": question})
print(response)