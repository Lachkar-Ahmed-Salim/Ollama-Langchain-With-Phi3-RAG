from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import asyncio


async def run_chain():
    llm = ChatOllama(model="phi3")
    prompt = ChatPromptTemplate.from_template("answer in {question} from the user")

    chain = prompt | llm | StrOutputParser()

    question = input("Enter your question: ") 
    chain.invoke({"question": question})

    chunks = []
    async for chunk in chain.astream(question):
        chunks.append(chunk)
        print(chunk, end="", flush=True)

# Run the async function
asyncio.run(run_chain())
