import os
import streamlit as st
from dotenv import load_dotenv

# --- Imports ---
# Cleaned up imports into a single, organized block.
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from vector_utils import load_vectorstore

# --- Load Environment Variables ---
# This ensures the HF_TOKEN is loaded from your .env file at the start.
load_dotenv()

# --- App Configuration & Constants ---
st.set_page_config(page_title="MediBot", page_icon="⚕️", layout="centered")

HUGGING_FACE_REPO_ID = "google/flan-t5-base"

CUSTOM_PROMPT_TEMPLATE = """You are a knowledgeable medical assistant specializing in providing clear, accurate information from medical sources.

Based on the following medical context, answer the user's question in a structured, easy-to-understand format. Follow these guidelines:

1. **Structure your response**: Use bullet points, numbered lists, or clear headings to organize information when appropriate.
2. **Be concise yet comprehensive**: Provide all relevant details from the context without unnecessary verbosity.
3. **Use appropriate medical terminology**: Explain technical terms simply when needed.
4. **Cite sources**: Reference the provided context when possible.
5. **Be honest**: If the context doesn't contain enough information to fully answer the question, clearly state what you can and cannot answer.
6. **Ensure complete responses**: Always provide a clear introduction to your answer and end with key takeaways or summary points. Do not cut off mid-sentence.

Context: {context}
Question: {question}

Structured Answer:"""

# --- Caching Functions for Performance ---
# Caching the vector store loading.
@st.cache_resource
def get_vectorstore():
    """Loads the FAISS vector store from the local path."""
    db = load_vectorstore()
    if db is None:
        st.error("Vector store not found. Please run `create_memory_for_llm.py` to build it.")
    return db

# Caching the LLM loading for significant speed improvement.
@st.cache_resource
def load_llm():
    """Loads the Hugging Face model using local pipeline."""
    try:
        llm = HuggingFacePipeline.from_model_id(
            model_id=HUGGING_FACE_REPO_ID,
            task="text2text-generation",
            device=-1,  # Use CPU
            model_kwargs={"temperature": 0.1, "max_length": 2048}
        )
        return llm
    except Exception as e:
        st.error(f"Failed to load LLM: {str(e)}")
        return None

# Caching the retrieval chain creation.
@st.cache_resource
def load_qa_chain(_vectorstore, _llm):
    """Creates and returns the QA retrieval chain."""
    retriever = _vectorstore.as_retriever(search_kwargs={"k": 3})
    prompt = PromptTemplate(template=CUSTOM_PROMPT_TEMPLATE, input_variables=["context", "question"])
    chain = prompt | _llm | StrOutputParser()
    return {"retriever": retriever, "chain": chain}

# --- Main Streamlit App Logic ---
def main():
    st.title("⚕️ Ask MediBot")
    st.markdown("Your personal assistant for medical questions, powered by local data.")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Load the components once and handle potential loading errors
    vectorstore = get_vectorstore()
    llm = load_llm()
    
    # *** THIS IS THE MAIN LOGIC FIX ***
    # We proceed only if BOTH the vectorstore AND the llm loaded successfully.
    if vectorstore and llm:
        qa_chain = load_qa_chain(vectorstore, llm)

        # React to user input
        if prompt := st.chat_input("Ask a question, e.g., 'What are the symptoms of diabetes?'"):
            # Display user message
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.spinner("MediBot is thinking..."):
                try:
                    # Retrieve context documents
                    docs = qa_chain["retriever"].invoke(prompt)
                    
                    # Format context and invoke chain
                    context = "\n".join([doc.page_content for doc in docs])
                    result = qa_chain["chain"].invoke({"context": context, "question": prompt})
                    
                    # Display assistant response
                    with st.chat_message("assistant"):
                        st.markdown(result)
                        with st.expander("View Sources"):
                            # Nicely format the source documents
                            for i, doc in enumerate(docs):
                                page_num = doc.metadata.get("page", "N/A") if hasattr(doc, "metadata") else "N/A"
                                source_file = doc.metadata.get("source", "Unknown") if hasattr(doc, "metadata") else "Unknown"
                                st.info(f"**Source {i+1}** | Page: {page_num} \n\n{doc.page_content}")

                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": result})

                except Exception as e:
                    st.error(f"An error occurred while generating the response: {str(e)}")
    else:
        st.warning("The chatbot is not available due to a configuration issue. Please check the logs.")

if __name__ == "__main__":
    main()
