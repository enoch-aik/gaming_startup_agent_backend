# 🧠 GenAI Chat Agent Documentation
**Project Title**: AI Agent for Assisting Experimentation in Video Game Startups  
**Developer**: Enoch Aikpokpodion  
**Python Version**: 3.11  
**Focus Area**: Technical Prototyping in Adventure Games

---

## 📘 Overview

This AI-powered chat agent is designed to assist video game startups, especially those in the **Adventure game** genre with technical prototyping during early-stage experimentation. It uses ChatGPT 4-0 with a **Retrieval-Augmented Generation (RAG)** architecture to provide startup teams with contextually relevant, up-to-date insights from curated gaming data sources like blogs, forums, and review sites.

---

## 🔍 Capabilities

- **Query-Based Assistance** – It responds to questions about gameplay ideas, mechanics, and technical feasibility.
- **Idea Refinement** – It helps brainstorm and enhance early-stage game concepts.
- **Knowledge Retrieval** – It pulls up-to-date information from game dev forums, Steam reviews, Gamasutra blogs, and other curated sources.
- **Context Memory** – Maintains short-term conversation memory to improve coherence.

---

## 🛠 Architecture Overview

### Backend Stack

- **Language**: Python 3.11
- **Framework**: [LangChain](https://www.langchain.com/)
- **Embedding Model**: `text-embedding-3-small` (OpenAI)
- **Vector Store**: Pinecone
- **LLM Generator**: GPT (via OpenAI)
- **RAG Pipeline**: Retrieves relevant chunks and injects them into prompt context
- **Hosting**: Heroku
- **Database**: Firebase Firestore (encrypted)
- **Auth**: Firebase Auth (Anonymous)

### Frontend Stack
Link to frontend repository: [Frontend](https://github.com/enoch-aik/gaming_startup_ai_agent)
- **Framework**: Flutter Web
- **Features**:
    - Chat UI
    - Anonymous authentication
    - Cross-platform support

---

## ▶️ How to Run It Locally

### 1. Clone the Repository
```bash
git clone https://github.com/enoch-aik/gaming_startup_agent_backend
cd gaming_startup_agent_backend
```


### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables
- Create a `.env` file in the root directory and add your OpenAI API key, Pinecone API key, and Firebase credentials. Example:
```bash
export OPEN
OPENAI_API_KEY='your_openai_api_key'
export PINECONE_API_KEY='your_pinecone_api_key'
export FIREBASE_CREDENTIALS='path_to_your_firebase_credentials.json'
```
### 4. Run the Application
```bash
python app.py
```


### 5. Access the Chat Interface
Open your browser and navigate to `http://localhost:5000` to interact with the AI agent.

### 6. Deploy to Heroku (Optional)

```bash
heroku create gaming-startup-agent
git push heroku main
```

