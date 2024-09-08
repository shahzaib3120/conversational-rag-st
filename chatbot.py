import os
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
import dotenv
import asyncio
from operator import itemgetter
import streamlit as st

st.set_page_config(page_title="Chatbot", page_icon="")
st.title("RAG Chatbot")

"""
This is a sample title
"""
dotenv.load_dotenv(".env")

# Set up memory
msgs = StreamlitChatMessageHistory(key="langchain_messages")
if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you?")

view_messages = st.expander("View the message contents in session state")

# Set up the LangChain, passing in Message History
system_prompt = (
    "You are an assistant for question-answering tasks."
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)

# ChatGPT
llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

# Gemini
# llm = ChatGoogleGenerativeAI(
#     model="gemini-1.5-pro",
#     temperature=0,
#     max_tokens=None,
#     timeout=None,
#     max_retries=2,
#     # other params...
# )

# NOTE: Rebuilt the vector store when changing the embedding model

# Free SentenceTransformer model for embeddings
# embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
embedding = OpenAIEmbeddings(model="text-embedding-3-small")

persist_directory = "vector_store"

# Initialize Chroma vector store for document embeddings
# and create retriever from the vector store
text_loader_kwargs = {"autodetect_encoding": True}
txt_loader = DirectoryLoader(
    "docs",
    glob="./*.txt",
    loader_cls=TextLoader,
    show_progress=True,
    loader_kwargs=text_loader_kwargs,
)

loader = txt_loader
if not os.path.exists(persist_directory):
    print("Creating new vector database")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    os.makedirs(persist_directory)
    vectordb = Chroma.from_documents(
        documents=texts, embedding=embedding, persist_directory=persist_directory
    )
else:
    print("Loading existing vector database")
    vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
retriever = vectordb.as_retriever()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


print("Retriever initialized")

chain = (
    {
        "context": itemgetter("question") | retriever | format_docs,
        "question": itemgetter("question"),
        "history": itemgetter("history"),
    }
    | prompt
    | llm
)
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: msgs,
    input_messages_key="question",
    history_messages_key="history",
)

# Render current messages from StreamlitChatMessageHistory
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)


async def handle_prompt(prompt: str):
    with st.chat_message("human"):
        st.markdown(prompt)
    with st.chat_message("ai"):
        # Note: new messages are saved to history automatically by Langchain during run
        config = {"configurable": {"session_id": "any"}}
        # response = ""
        # async for chunk in chain_with_history.astream({"question": prompt}, config):
        #     if response_chunk := chunk:
        #         response += response_chunk.content
        #         st.write(response)

        response = chain_with_history.invoke({"question": prompt}, config)
        st.write(response.content)


# If user inputs a new prompt, generate and draw a new response
if prompt := st.chat_input():
    asyncio.run(handle_prompt(prompt))


# Draw the messages at the end, so newly generated ones show up immediately
with view_messages:
    """
    Message History initialized with:
    ```python
    msgs = StreamlitChatMessageHistory(key="langchain_messages")
    ```

    Contents of `st.session_state.langchain_messages`:
    """
    view_messages.json(st.session_state.langchain_messages)
