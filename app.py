import streamlit as st
from ai_notes_generator import get_summarizer, generate_notes, clean_text

st.title("AI Notes Generator")

uploaded = st.file_uploader("Upload a text file", type=['txt'])
model_name = st.text_input("Summarization model", value="sshleifer/distilbart-cnn-12-6")

if uploaded:
    raw = uploaded.getvalue().decode('utf-8')
    st.text_area("Raw notes", value=raw, height=200)
    if st.button("Generate Notes"):
        with st.spinner("Loading model and generating..."):
            summarizer = get_summarizer(model_name)
            md = generate_notes(raw, summarizer)
        st.markdown("### Formatted Notes")
        st.markdown(md)
