import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Local embedding model (runs on your machine, no API key, no rate limits).
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _get_groq_api_key():
    """Read the Groq API key from Streamlit secrets (cloud) or env/.env (local)."""
    key = os.getenv("GROQ_API_KEY")
    if not key:
        try:
            key = st.secrets["GROQ_API_KEY"]
        except Exception:
            key = None
    return key


GROQ_API_KEY = _get_groq_api_key()
if GROQ_API_KEY:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY
else:
    st.error("GROQ_API_KEY is not set. Add it to a .env file locally, "
             "or to app secrets when deploying on Streamlit Cloud. "
             "Get a free key at https://console.groq.com")


@st.cache_resource(show_spinner="Loading embedding model...")
def get_embeddings():
    """Load the local HuggingFace embedding model once and reuse it."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text


def get_text_chunks(text):
    # Smaller chunks keep each retrieved piece of context compact, which keeps
    # the final prompt well under the model's tokens-per-minute limit.
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    """Build a FAISS index using local embeddings (no API calls, no rate limits)."""
    total = len(text_chunks)
    if total == 0:
        st.error("No text could be extracted from the uploaded PDF(s).")
        return

    embeddings = get_embeddings()
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGroq(model="openai/gpt-oss-120b", temperature=0.3)

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    return model, prompt


# Cap the total context sent to the model. Groq's free tier allows 8000
# tokens/minute; ~4 chars per token means ~24000 chars is a safe ceiling
# once the prompt template and the model's reply are accounted for.
MAX_CONTEXT_CHARS = 12000


def user_input(user_question):
    embeddings = get_embeddings()

    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    # Retrieve the top matching chunks only.
    docs = new_db.similarity_search(user_question, k=4)

    model, prompt = get_conversational_chain()

    # Join retrieved chunks but stop once we hit the context size cap, so the
    # prompt never exceeds the free-tier tokens-per-minute limit (413 error).
    context_parts = []
    used = 0
    for doc in docs:
        piece = doc.page_content
        if used + len(piece) > MAX_CONTEXT_CHARS:
            piece = piece[: max(0, MAX_CONTEXT_CHARS - used)]
        if piece:
            context_parts.append(piece)
            used += len(piece)
        if used >= MAX_CONTEXT_CHARS:
            break

    context = "\n\n".join(context_parts)
    message = prompt.format(context=context, question=user_question)
    response = model.invoke(message)

    answer = _extract_text(response)
    print(answer)
    st.write("Reply: ", answer)


def _extract_text(response):
    """Normalize a chat response into plain text.

    Chat responses may return content as a plain string or as a list of parts
    (e.g. [{"type": "text", "text": "..."}]).
    """
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(part.get("text", ""))
            else:
                parts.append(str(part))
        return "".join(parts)
    return str(content)


def main():
    st.set_page_config("Multi PDF Chatbot", page_icon=":scroll:")
    st.header("Multi-PDF's 📚 - Chat Agent 🤖 ")

    user_question = st.text_input("Ask a Question from the PDF Files uploaded .. ✍️📝")

    if user_question:
        if not os.path.exists("faiss_index"):
            st.warning("Please upload and process PDF files first (use the sidebar).")
        else:
            user_input(user_question)

    with st.sidebar:

        st.image("img/Robot.jpg")
        st.write("---")

        st.title("📁 PDF File's Section")
        pdf_docs = st.file_uploader("Upload your PDF Files & \n Click on the Submit & Process Button ", accept_multiple_files=True)
        if st.button("Submit & Process"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF file.")
            else:
                with st.spinner("Processing..."):  # user friendly message.
                    raw_text = get_pdf_text(pdf_docs)  # get the pdf text
                    text_chunks = get_text_chunks(raw_text)  # get the text chunks
                    get_vector_store(text_chunks)  # create vector store
                    st.success("Done")

        st.write("---")
        st.image("img/Robot.jpg")
        st.write("AI App created by @ Praveen Kumar")  # add this line to display the image

    st.markdown(
        """
        <div style="position: fixed; bottom: 0; left: 0; width: 100%; background-color: #0E1117; padding: 15px; text-align: center;">
            © <a href="https://github.com/praveenrathod12" target="_blank">Praveen Kumar</a> | Made with ❤️
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
