from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import os
from pathlib import Path

cromadb = Path(__file__).parent / "Database"


load_dotenv()

# model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.2,
)

embedding = HuggingFaceEmbeddings(model=os.getenv("EMBADDING_MODEL"))

vectorStore = Chroma(persist_directory=str(cromadb), embedding_function=embedding)

# retriever = vectorStore.as_retriever(
#     search_type="mmr", search_kwargs={"k": 10, "fetch_k": 20, "lambda_mult": 0.7}
# )
retriever = vectorStore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

Template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
              you are a helpfull Ai Assistant use only the
               provided the context to answer the question 
               if the answer is not presnt in the context 
               say 'i could not find the answer in the document'
    
                 """,
        ),
        (
            "human",
            """
               content : {context},
               Question : {user_input}
                 """,
        ),
    ]
)


while True:
    user_input = input("\n Ask Question : ")
    if user_input == "Exit" or user_input.lower() == "exit":
        break
    docs = retriever.invoke(user_input)
    print(f"length of the docs : {len(docs)}")
    for i, doc in enumerate(docs, start=1):
        print(f"\n--- Chunk {i} ---")
        print(doc.page_content)

    if not docs:
        print("I could not find the answer in the document.")
        continue
    context = "\n\n".join([doc.page_content for doc in docs])

    final_prompt = Template.invoke({"context": context, "user_input": user_input})

    response = model.stream(final_prompt)
    for chunk in response:
        print(chunk.content, end="", flush=True)
