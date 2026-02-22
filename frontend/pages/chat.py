import streamlit as st
import requests
import json
import uuid

API_URL = 'http://api:8000'

st.title('💬 Agent Chat')

# Initialise session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'thread_id' not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if 'pending_approval' not in st.session_state:
    st.session_state.pending_approval = False
if 'agent_steps' not in st.session_state:
    st.session_state.agent_steps = []

# Thread ID display (for debugging / resuming sessions)
with st.expander('Session Info'):
    st.text(f'Thread ID: {st.session_state.thread_id}')
    if st.button('Start New Session'):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.pending_approval = False
        st.session_state.agent_steps = []
        st.rerun()

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])

# Human-in-the-loop approval UI
if st.session_state.pending_approval:
    st.warning('⚠️ The agent has paused and is waiting for your approval.')
    col1, col2 = st.columns(2)
    with col1:
        if st.button('✅ Approve Report Generation', type='primary'):
            resp = requests.post(f'{API_URL}/agent/approve', json={
                'thread_id': st.session_state.thread_id,
                'decision': 'approved',
                'reviewer_name': 'Auditor'
            })
            data = resp.json()
            st.session_state.messages.append({
                'role': 'assistant',
                'content': data.get('response', 'Report generation approved.')
            })
            st.session_state.pending_approval = False
            st.rerun()
    with col2:
        if st.button('❌ Reject', type='secondary'):
            requests.post(f'{API_URL}/agent/approve', json={
                'thread_id': st.session_state.thread_id,
                'decision': 'rejected',
                'reviewer_name': 'Auditor'
            })
            st.session_state.messages.append({
                'role': 'assistant', 'content': 'Report generation rejected.'
            })
            st.session_state.pending_approval = False
            st.rerun()

# Chat input
if prompt := st.chat_input('Ask about audit findings or give the agent a task...'):
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    with st.chat_message('user'):
        st.markdown(prompt)

    # Stream agent response
    with st.chat_message('assistant'):
        steps_placeholder = st.empty()
        response_placeholder = st.empty()
        steps_so_far = []
        final_response = ''
        needs_approval = False

        with requests.post(
            f'{API_URL}/agent/stream',
            json={
                'message': prompt,
                'thread_id': st.session_state.thread_id,
                'require_approval': True
            },
            stream=True,
            timeout=60
        ) as resp:
            for line in resp.iter_lines():
                if line and line.startswith(b'data: '):
                    data = json.loads(line[6:])
                    steps_so_far.extend(data.get('steps', []))
                    if steps_so_far:
                        steps_placeholder.info(
                            'Agent steps: ' + ' → '.join(steps_so_far[-3:])
                        )
                    if data.get('response'):
                        final_response = data['response']
                        response_placeholder.markdown(final_response)
                    if data.get('needs_approval'):
                        needs_approval = True

        st.session_state.agent_steps = steps_so_far
        if needs_approval:
            st.session_state.pending_approval = True
            st.session_state.messages.append({
                'role': 'assistant',
                'content': '⏸️ Paused — waiting for human approval...'
            })
        elif final_response:
            st.session_state.messages.append({
                'role': 'assistant', 'content': final_response
            })
        st.rerun()
