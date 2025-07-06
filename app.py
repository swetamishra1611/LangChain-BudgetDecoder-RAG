# app.py

import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# --- 1. UI Configuration ---
st.set_page_config(page_title="RAG Q&A on Union Budget 2024", layout="wide")
st.title("📄 RAG Q&A on Union Budget 2024")
st.markdown("""
<style>
    .stSpinner > div > div {
        border-top-color: #ff4b4b;
    }
</style>
""", unsafe_allow_html=True)


# --- 2. Caching the RAG Chain Setup ---
@st.cache_resource(show_spinner="Setting up the RAG chain... This may take a few minutes.")
def setup_rag_chain():
    """
    Sets up the entire RAG pipeline and returns the chain.
    This function is cached to avoid reloading models and data on each interaction.
    """
    # a. Load Data
    urls = [
        'https://www.livemint.com/economy/budget-2024-key-highlights-live-updates-nirmala-sitharaman-infrastructure-defence-income-tax-modi-budget-23-july-11721654502862.html',
        'https://cleartax.in/s/budget-2024-highlights',
        'https://www.hindustantimes.com/budget',
        'https://economictimes.indiatimes.com/news/economy/policy/budget-2024-highlights-india-nirmala-sitharaman-capex-fiscal-deficit-tax-slab-key-announcement-in-union-budget-2024-25/articleshow/111942707.cms?from=mdr'
    ]
    loader = WebBaseLoader(urls)
    try:
        data = loader.load()
    except Exception as e:
        st.error(f"Failed to load documents from URLs: {e}")
        return None

    # b. Split Data
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(data)
    if not docs:
        st.error("Could not split the documents. Check if the URLs are accessible and contain text.")
        return None

    # c. Setup Embeddings and Vector Store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # d. Setup LLM
    # Note: Running LLMs locally requires significant RAM/VRAM.
    # gpt-neo-1.3B is a smaller model suitable for CPU or modest GPUs.
    model_name = "EleutherAI/gpt-neo-1.3B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Use 'cpu' if you don't have a compatible GPU or CUDA is not set up
    device = 0 if torch.cuda.is_available() else -1 

    text_generation_pipeline = pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        temperature=0.2,
        repetition_penalty=1.1,
        return_full_text=True,
        max_new_tokens=400,
        device=device
    )
    llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

    # e. Create Prompt and RAG Chain
    prompt_template = """
    Answer the question based only on the following context. If the answer is not in the context, say "I don't have enough information from the provided documents".

    CONTEXT:
    {context}

    QUESTION:
    {question}

    ANSWER:
    """
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template,
    )

    rag_chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()
    
    return rag_chain

# --- 3. Main Application Logic ---

# Get the RAG chain (it will be created and cached on the first run)
rag_chain = setup_rag_chain()

if rag_chain:
    # Get user input
    user_question = st.text_input("Ask a question about the Union Budget 2024:")

    if user_question:
        with st.spinner("Generating answer..."):
            try:
                response = rag_chain.invoke(user_question)
                # Cleaning up the response to remove potential artifacts
                # The prompt structure might be repeated in the output by some models
                if "ANSWER:" in response:
                    response = response.split("ANSWER:")[1].strip()
                st.markdown("### Answer")
                st.write(response)
            except Exception as e:
                st.error(f"An error occurred while generating the answer: {e}")
else:
    st.warning("The RAG chain could not be initialized. Please check the logs and URLs.")