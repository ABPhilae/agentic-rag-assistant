import streamlit as st

st.title('🔍 Agent Execution Trace')
st.markdown('See exactly what the agent did step by step in your last conversation.')

if 'agent_steps' not in st.session_state or not st.session_state.agent_steps:
    st.info('No agent steps recorded yet. Start a conversation in Agent Chat.')
else:
    st.markdown('### Steps taken in last agent run')
    for i, step in enumerate(st.session_state.agent_steps, 1):
        st.markdown(f'**{i}.** {step}')
    if st.session_state.get('pending_approval'):
        st.warning('Agent is currently paused — waiting for human approval.')
