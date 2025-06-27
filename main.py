import streamlit as st
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from openai import OpenAI
import tempfile

# Load OpenAI client using secret key
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

st.set_page_config(page_title="📘 PDF RAG Chatbot", layout="centered")
st.title("📘 PDF RAG Chatbot with Qdrant")

# Initialize session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "messages" not in st.session_state:
    st.session_state.messages = []  # to hold full conversation

uploaded_file = st.file_uploader("📤 Upload your PDF file", type="pdf")

if uploaded_file:
    with st.spinner("Processing..."):

        # Save uploaded file to a temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_pdf_path = Path(tmp_file.name)

        # Load and split
        loader = PyPDFLoader(str(temp_pdf_path))
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        split_docs = splitter.split_documents(docs)

        for doc in split_docs:
            doc.metadata["source"] = uploaded_file.name

        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-large",
            api_key=st.secrets["OPENAI_API_KEY"]
        )

        vector_store = Qdrant.from_documents(
            documents=split_docs,
            embedding=embeddings,
            url=st.secrets["QDRANT_CLOUD_URL"],
            api_key=st.secrets["QDRANT_API_KEY"],
            collection_name="learning_vectors",
            force_recreate=True
        )

        st.session_state.vector_store = vector_store
        st.success("✅ Processing Done! You can now ask questions below.")

# Display past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Ask new question
if st.session_state.vector_store:
    query = st.chat_input("Ask a question based on the PDF")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})

        with st.chat_message("user"):
            st.markdown(query)

        with st.spinner("🔍 Searching context..."):
            results = st.session_state.vector_store.similarity_search(query)
            context = "\n\n\n".join([
                f"Page Content: {doc.page_content}\nPage Number: {doc.metadata.get('page_label', '?')}\nFile Location: {doc.metadata['source']}"
                for doc in results
            ])

        SYSTEM_PROMPT = f"""
        You are a helpful AI Assistant who answers user queries based on the available context
        retrieved from a PDF file along with page contents and page numbers.

        You should only answer based on the following context and help navigate the user
        to the right page to explore more.

        Context:
        {context}
        """

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ] + st.session_state.messages

        with st.chat_message("assistant"):
            response = client.chat.completions.create(
                model="gpt-4.1",
                messages=messages
            )
            ai_reply = response.choices[0].message.content
            st.markdown(ai_reply)

        st.session_state.messages.append({"role": "assistant", "content": ai_reply})

# Footer
st.markdown(
    """
    <hr style="margin-top: 2rem; margin-bottom: 1rem;">
    <div style='text-align: center; color: gray; font-size: 0.9rem'>
        🚀 Built with ❤️ by <b>Manunjay Bhardwaj</b> | Powered by LangChain, OpenAI & Qdrant
    </div>
    """,
    unsafe_allow_html=True
)
