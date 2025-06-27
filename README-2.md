
# 📘 PDF RAG Chatbot using Streamlit, LangChain, OpenAI & Qdrant Cloud

This project is a fully functional PDF-based **RAG (Retrieval-Augmented Generation)** chatbot built using **Streamlit**, **LangChain**, **OpenAI**, and **Qdrant Cloud**. Upload any PDF and ask questions — the app finds relevant context using vector search and answers using GPT.

---

## 🚀 Features

- ✅ Upload any PDF file
- ✂️ Auto text chunking & overlap with LangChain
- 🧠 Embedding generation via OpenAI (`text-embedding-3-large`)
- 🔍 Vector search using Qdrant Cloud
- 🤖 GPT-4-based conversational chatbot with chat history
- ☁️ Deployed on Streamlit Cloud

---

## 📦 Tech Stack

- [Streamlit](https://streamlit.io/) – UI Framework
- [LangChain](https://www.langchain.com/) – Chunking & RAG pipeline
- [OpenAI API](https://platform.openai.com/) – GPT-4 & Embeddings
- [Qdrant Cloud](https://cloud.qdrant.io) – Vector DB for semantic search

---

## 📁 Project Structure

```
RAG_CHAT/
├── app.py               # Main Streamlit app
├── requirements.txt     # All dependencies
├── .streamlit/
│   └── secrets.toml     # API keys (on cloud)
└── README.md            # This file
```

---

## 🔑 Setup Instructions

### 1. Clone the repo

```bash
git clone https://github.com/your-username/pdf-rag-chatbot.git
cd pdf-rag-chatbot
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Create `.env` (for local testing)

```env
QDRANT_CLOUD_URL="https://your-cluster-name.qdrant.tech"
QDRANT_API_KEY="your-qdrant-api-key"
OPENAI_API_KEY="your-openai-key"
```

### 4. Run the app locally

```bash
streamlit run app.py
```

---

## 🌐 Deployment to Streamlit Cloud

1. Push your repo to GitHub
2. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud) and connect your repo
3. In `Secrets` tab, add:

```toml
QDRANT_CLOUD_URL = "https://your-cluster-name.qdrant.tech"
QDRANT_API_KEY = "your-qdrant-api-key"
OPENAI_API_KEY = "your-openai-key"
```

4. Deploy and enjoy!

---

## 👨‍💻 Author

Built with ❤️ by **Manunjay Bhardwaj**  
Follow me on [LinkedIn](https://www.linkedin.com/in/manunjaybhardwaj)

