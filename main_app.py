import streamlit as st
import glob
import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings

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
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("Please configure GOOGLE_API_KEY in Streamlit secrets!")
    st.stop()

# 3. Create RAG Pipeline
@st.cache_resource
def load_rag_pipeline():
    docs = []
    pdf_files = glob.glob("dataset/**/*.pdf", recursive=True)
    docx_files = glob.glob("dataset/**/*.docx", recursive=True)
    
    if not pdf_files and not docx_files:
        st.warning("No documents found in 'dataset' folder. Please upload PDF or DOCX files to GitHub.")
        st.stop()

    for file in pdf_files:
        try:
            docs.extend(PyPDFLoader(file).load())
        except Exception as e:
            st.error(f"Error loading {file}: {e}")
            
    for file in docx_files:
        try:
            docs.extend(Docx2txtLoader(file).load())
        except Exception as e:
            st.error(f"Error loading {file}: {e}")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vectorstore = FAISS.from_documents(splits, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

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
with st.spinner('Preparing AI & Documents... This may take a moment on first run.'):
    try:
        rag_chain = load_rag_pipeline()
    except Exception as e:
        st.error(f"Failed to initialize RAG pipeline: {e}")
        st.stop()

# 5. Example Questions Section
st.markdown("#### 💡 Example Questions:")
examples = [
    "What is the MLii Ebook Fund?",
    "Who is eligible to apply for this fund?",
    "What documents are required for the application?"
]

# Create a button for the sample question
col1, col2, col3 = st.columns(3)
with col1:
    if st.button(examples[0], use_container_width=True):
        st.session_state.example_selected = examples[0]
with col2:
    if st.button(examples[1], use_container_width=True):
        st.session_state.example_selected = examples[1]
with col3:
    if st.button(examples[2], use_container_width=True):
        st.session_state.example_selected = examples[2]

st.divider()

# 6. Chat UI
if "messages" not in st.session_state:
    st.session_state.messages = []

# Show chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle input (receive from the typing field or from the Example button).
user_input = st.chat_input("Ask anything about the MLii Ebook Fund here...")

# If the user presses the Example button, use that value instead.
if "example_selected" in st.session_state:
    user_input = st.session_state.example_selected
    del st.session_state.example_selected # Delete the value after it's been used to prevent it from being used repeatedly.ำ

if user_input:
    # 1. Show user questions
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. Process and display the answer from the AI.
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = rag_chain.invoke({"input": user_input})
                answer = response['answer']
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Error generating response: {e}")
