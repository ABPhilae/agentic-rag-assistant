import streamlit as st

st.set_page_config(
    page_title='Agentic Audit Assistant',
    page_icon='🏦',
    layout='wide',
    initial_sidebar_state='expanded'
)

st.sidebar.title('🏦 Audit Agent')
st.sidebar.markdown('---')
st.sidebar.markdown('**Navigation**')
st.sidebar.page_link('pages/chat.py', label='💬 Agent Chat')
st.sidebar.page_link('pages/upload.py', label='📤 Upload Documents')
st.sidebar.page_link('pages/status.py', label='🔍 Agent Trace')
st.sidebar.markdown('---')
st.sidebar.caption('Phase 4 Project 1 | LangGraph + NeMo')

# Default landing page
st.title('Agentic RAG Assistant')
st.markdown("""
Welcome to your intelligent audit assistant. Unlike a basic search tool,
this agent **reasons and acts**: it decides what to search, checks compliance,
verifies deadlines, and asks for your approval before generating sensitive reports.

### How to use
1. **Upload** your audit documents in 📤 Upload Documents
2. **Ask** questions or give tasks in 💬 Agent Chat
3. **Review** what the agent did in 🔍 Agent Trace
""")
