

import streamlit as st
from dotenv import load_dotenv
import os
import shutil
import hashlib

from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()

st.set_page_config(page_title="AI Document Assistant", page_icon="📚")

st.title("📚 AI Document Assistant")
st.write("Upload a PDF and ask questions using your AI Document Assistant.")


# -------------------- Original Configuration --------------------

PDF_PATH = os.path.join("document_loading", "xyz.pdf")
VECTORSTORE_PATH = "chroma-db"


# -------------------- Learning Mode Selection --------------------

st.subheader("Choose your AI Document Assistant Mode")

prompt_choice = st.radio(
    "Select a mode:",
    [
        "1. 😊 Friendly Mode",
        "2. 📚 Exam Mode",
        "3. 🔥 Strict Mode"
    ]
)

match prompt_choice:
    case "1. 😊 Friendly Mode":
        mode = "You are a friendly, patient, and supportive AI tutor. Make the student feel comfortable asking questions, even if they lose focus or need repeated explanations. Start naturally with reassuring phrases such as `No worries! Let's understand this step by step.` Explain concepts in clear, beginner-friendly language, breaking difficult ideas into manageable parts. Use a warm, conversational tone and guide the student patiently without making them feel judged. Adapt your explanation when the student says they do not understand. Use only the provided PDF context."

    case "2. 📚 Exam Mode":
        mode = "You are an exam-revision tutor assisting a student who has only one hour left before an exam. Be highly focused, concise, and exam-oriented. Start with phrases such as `Time is limited. Focus on these key points.` Present answers using numbered points, essential definitions, important keywords, and formulas where relevant. Highlight what the student should memorize and finish with a clearly labeled `LAST-MINUTE REVISION` section containing the most important takeaways. Avoid lengthy introductions and unnecessary explanations. Prioritize information from the provided PDF. Never invent missing information; state when the answer cannot be found in the document."

    case "3. 🔥 Strict Mode":
        mode = "You are a strict, disciplined, and results-oriented tutor. Maintain a firm, direct, no-nonsense tone. Start with strict phrases example: `Focus and be an attentive student !`, `ok so Stop being lazy and focus on this topic now` Be direct about misunderstandings and insist on conceptual clarity rather than superficial memorization. Explain the material in a structured manner, identify the central concept, and point out what the student must understand before moving forward. Avoid excessive reassurance, casual conversation, and unnecessary filler. Be demanding about learning, but never insult, humiliate, or belittle the student. Use only the provided PDF context and clearly state when the answer is unavailable."


# -------------------- Original Embedding Model --------------------

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


embeddings = get_embeddings()


# -------------------- PDF Upload --------------------

uploaded_file = st.file_uploader(
    "Upload your PDF document",
    type=["pdf"]
)

if uploaded_file:

    os.makedirs(os.path.dirname(PDF_PATH), exist_ok=True)

    pdf_bytes = uploaded_file.getvalue()
    pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()

    # Save the uploaded PDF at the same path used by your original code
    if st.session_state.get("pdf_hash") != pdf_hash:

        with open(PDF_PATH, "wb") as pdf_file:
            pdf_file.write(pdf_bytes)

        st.session_state.pdf_hash = pdf_hash
        st.session_state.pop("multi_query_retriever", None)
        st.session_state.pop("chat_history", None)
        st.session_state.pop("messages", None)

    st.success(f"PDF saved successfully: {PDF_PATH}")

    if st.button("Create Vector Database", type="primary"):

        with st.spinner("Loading PDF and creating vector database..."):

            try:
                # Original document loading logic
                data = PyPDFLoader(PDF_PATH)
                docs = data.load()

                # Original text splitting logic
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=10
                )

                texts = text_splitter.split_documents(docs)

                if not texts:
                    st.error("No readable text was found in the PDF.")
                    st.stop()

                # Remove the previous database so that documents
                # from different uploaded PDFs do not get mixed.
                if os.path.exists(VECTORSTORE_PATH):
                    shutil.rmtree(VECTORSTORE_PATH)

                # Original vector store creation logic
                vectorstore = Chroma.from_documents(
                    documents=texts,
                    embedding=embeddings,
                    persist_directory=VECTORSTORE_PATH
                )

                # Original retriever configuration
                retriever = vectorstore.as_retriever(
                    search_kwargs={"k": 4}
                )

                # Original Groq model
                model = ChatGroq(
                    model="openai/gpt-oss-120b"
                )

                # Original MultiQueryRetriever
                multi_query_retriever = MultiQueryRetriever.from_llm(
                    retriever=retriever,
                    llm=model
                )

                st.session_state.multi_query_retriever = (
                    multi_query_retriever
                )

                st.session_state.chat_history = []
                st.session_state.messages = []

                st.success("Vector database created successfully!")
                st.rerun()

            except Exception as e:
                st.error(f"Error creating vector database: {e}")


# -------------------- Original Prompt --------------------

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """{system_message}

            Use ONLY the provided context to answer the question.
            If the answer is not present in the context, say: "I could not find the answer in the document."
            However first understand what the student is asking, if he/she is asking any general questiont like how are you, who are you, what i syour name etc etc then answer accordingly, but whenever he/she is asking any question related to context then answer strictly accordingly to the context given and say "I could not find the answer in the document." for cases where the answer is not found.
            """
        ),
        (
            "human",
            """Previous Conversation:{chat_history}
            Retrieved PDF Context:{context}
            Current Question:{question}"""
        )
    ]
)


# -------------------- Chat Interface --------------------

if "multi_query_retriever" in st.session_state:

    st.divider()
    st.subheader("💬 Ask Questions From Your PDF")

    # Display existing conversation
    for message in st.session_state.get("messages", []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    query = st.chat_input("Enter your question...")

    if query:

        st.session_state.messages.append(
            {"role": "user", "content": query}
        )

        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                try:
                    # Original retrieval logic
                    docs = st.session_state.multi_query_retriever.invoke(
                        query
                    )

                    unified_context = "\n\n".join(
                        [doc.page_content for doc in docs]
                    )

                    # Original model creation
                    model = ChatGroq(
                        model="openai/gpt-oss-120b"
                    )

                    # Original prompt invocation
                    ready_prompt = prompt.invoke(
                        {
                            "context": unified_context,
                            "question": query,
                            "system_message": mode,
                            "chat_history": st.session_state.chat_history
                        }
                    )

                    # Original LLM response generation
                    LLM_response = model.invoke(ready_prompt)

                    answer = LLM_response.content

                    st.markdown(answer)

                    # Original conversation history logic
                    st.session_state.chat_history.append(query)
                    st.session_state.chat_history.append(answer)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer
                        }
                    )

                except Exception as e:
                    st.error(f"An error occurred: {e}")

else:
    st.info(
        "Upload a PDF, click 'Create Vector Database', "
        "and then start asking questions."
    )
