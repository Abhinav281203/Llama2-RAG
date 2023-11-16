import streamlit as st
import torch
from glob import glob
from pathlib import Path
from llama_index.prompts.prompts import SimpleInputPrompt
from llama_index import (
    set_global_service_context,
    ServiceContext,
    VectorStoreIndex,
    download_loader,
)
from langchain.embeddings import HuggingFaceEmbeddings
from llama_index.embeddings import LangchainEmbedding
from transformers import AutoTokenizer, AutoModelForCausalLM
from llama_index.llms import HuggingFaceLLM

PyMuPDFReader = download_loader("PyMuPDFReader")
loader = PyMuPDFReader()
model_name = "meta-llama/Llama-2-7b-chat-hf"
auth_token = "*******************************"
system_prompt = """<s>[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.
Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. 
Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct.
If you don't know the answer to a question, please don't share false information. <</SYS>>
"""  # Llama2's official system prompt


@st.cache_resource
def model_tokenizer_embedder(model_name, auth_token):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, cache_dir="./model/", use_auth_token=auth_token
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir="./model/",
        use_auth_token=auth_token,
        torch_dtype=torch.float16,
        load_in_8bit=True,
    )

    embedding_llm = LangchainEmbedding(
        HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    )

    return tokenizer, model, embedding_llm


def load_documents(directory):
    documents = []
    for item_path in glob(directory + "*.pdf"):
        documents.extend(loader.load(file_path=Path(item_path), metadata=True))
    return documents


tokenizer, model, embedding_llm = model_tokenizer_embedder(model_name, auth_token)

query_wrapper_prompt = SimpleInputPrompt("{query_str} [/INST]")

llm = HuggingFaceLLM(
    context_window=4096,
    max_new_tokens=256,
    system_prompt=system_prompt,
    query_wrapper_prompt=query_wrapper_prompt,
    model=model,
    tokenizer=tokenizer,
)

service_context = ServiceContext.from_defaults(
    chunk_size=1024, llm=llm, embed_model=embedding_llm
)
set_global_service_context(service_context)

documents = load_documents("./documents/")
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()

st.title("Llama2 RAG")

prompt = st.text_input("Enter your prompt")
if prompt:
    response = query_engine.query(prompt)
    st.write(response.response)
