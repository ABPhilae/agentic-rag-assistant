import streamlit as st
import requests

API_URL = 'http://api:8000'
st.title('📤 Upload Audit Documents')
st.markdown('Upload PDF audit documents to index them into the vector database.')

uploaded_files = st.file_uploader(
    'Choose PDF files', type=['pdf'], accept_multiple_files=True
)
if uploaded_files and st.button('Upload and Index', type='primary'):
    for f in uploaded_files:
        with st.spinner(f'Indexing {f.name}...'):
            resp = requests.post(
                f'{API_URL}/documents/upload',
                files={'file': (f.name, f.getvalue(), 'application/pdf')}
            )
            if resp.status_code == 200:
                data = resp.json()
                st.success(f'✅ {f.name}: {data["chunks_indexed"]} chunks indexed')
            else:
                st.error(f'❌ {f.name}: Upload failed — {resp.text}')
