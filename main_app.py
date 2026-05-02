import streamlit as st
import glob
import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# 1. Page Configuration
st.set_page_config(page_title="MLii Ebook Fund Q&A", page_icon="📖")
st.title("📚 MLii Ebook Fund Q&A Assistant")
st.markdown("### BDA_Project2_Group 7")
st.markdown("""
**List of Members:**
- [6631501154] [Tanakorn Siriwongpaisal]
- [6631501160] [Burin Pattanachote]
- [6631501165] [Pattaphon Suriyacum]
- [6631501170] [Sitthichot Khuaenkhamsaen]
- [6631501189] [Phyo Thant Kyaw]
""")
st.divider()

# 2. Load API Key from Streamlit Secrets
# (When deploying to Streamlit Cloud, this key must be configured in the app settings)
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("Please configure GOOGLE_API_KEY in Streamlit secrets!")
    st.stop()

# 3. Create RAG Pipeline
@st.cache_resource
def load_rag_pipeline():
    # Load files (Make sure the path matches your GitHub repository)
    docs = []
    pdf_files = glob.glob("dataset/**/*.pdf", recursive=True)
    docx_files = glob.glob("dataset/**/*.docx", recursive=True)
    
    for file in pdf_files:
        docs.extend(PyPDFLoader(file).load())
    for file in docx_files:
        docs.extend(Docx2txtLoader(file).load())

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)

    # Use Google Embeddings API (Lightweight, saves memory)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_documents(splits, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # Use Gemini 1.5 Flash (Fast and efficient)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

    # System prompt with strict English constraints
    system_prompt = (
        "You are a helpful and knowledgeable assistant for the MLii Ebook Fund. "
        "Use the following pieces of retrieved context to answer the user's question. "
        "IMPORTANT RULES:\n"
        "1. You MUST answer STRICTLY in English. Do not include any Thai characters or words in your response.\n"
        "2. Answer ONLY what the user asks. DO NOT generate follow-up questions, and DO NOT generate the 'Human:' tag.\n"
        "3. If you don't know the answer, just say that you don't know.\n\n"
        "Context: {context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    qa_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, qa_chain)

# 4. Load the system
with st.spinner('Preparing AI & Documents...'):
    rag_chain = load_rag_pipeline()

# 5. Chat UI
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Ask anything about the MLii Ebook Fund here..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = rag_chain.invoke({"input": user_input})
            answer = response['answer']
            st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
