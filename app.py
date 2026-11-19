# app.py: a simple web page for CityLens, built with Streamlit
import streamlit as st          # builds the web page
from ask import answer          # the function from ask.py that finds chunks and asks Claude

# Page settings: the browser tab title and icon
st.set_page_config(page_title="CityLens", page_icon="🏙️")

# Title and a one-line description at the top of the page
st.title("CityLens")
st.write("Ask plain-English questions about City of Edmonton budgets and zoning rules. "
         "Every answer shows the pages it came from.")

# Example questions the user can click, so they know what to try
examples = [
    "Where does the city's money come from?",
    "What kinds of housing can be built in the RS zone?",
    "Does Edmonton require a minimum number of parking spaces?",
]
st.caption("Try an example:")
cols = st.columns(len(examples))
for col, ex in zip(cols, examples):
    if col.button(ex):                       # a button for each example
        st.session_state["question"] = ex    # clicking fills the question box

# The question box. Its text is stored under the key "question".
question = st.text_input("Your question", key="question",
                         placeholder="e.g. When did the new zoning bylaw take effect?")

if question:
    # Show a spinner while the search and Claude work
    with st.spinner("Searching the documents..."):
        text, chunks = answer(question)      # returns Claude's answer and the 5 chunks

    st.subheader("Answer")
    st.write(text)

    st.subheader("Sources")
    st.caption("These are the pages the search retrieved. Open one to check the answer.")
    for c in chunks:
        # An expandable box for each source chunk, labelled with file and page
        with st.expander(f"{c['source']}, page {c['page']}"):
            st.write(c["text"])

# A note at the bottom
st.caption("Based on public City of Edmonton documents. Check the source before relying on an answer.")