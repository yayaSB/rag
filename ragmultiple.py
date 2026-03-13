import tempfile
import streamlit as st
from PIL import Image, ImageEnhance
import pytesseract
import re
from PyPDF2 import PdfReader
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

PROMPT_TEMPLATE = """
Tu es un assistant RAG multimodal.

Réponds uniquement à partir du contexte fourni.

Si la question est courte, ambiguë, ou contient seulement un mot,
essaie quand même d’interpréter la demande à partir du contexte récupéré.

Si l'information existe dans le contexte, réponds clairement et de façon structurée.
Si l'information n'existe vraiment pas dans le contexte, dis exactement :
"Je ne trouve pas cette information dans les documents fournis."

Contexte :
{context}

Question :
{question}
"""

llm = ChatOpenAI(model="gpt-4o", temperature=0)


def preprocess_image_for_ocr(image):
    image = image.convert("L")
    image = ImageEnhance.Contrast(image).enhance(3)
    image = image.resize((image.width * 2, image.height * 2))
    image = image.point(lambda x: 0 if x < 150 else 255, "1")
    return image


def extract_dates(text):
    text = text.replace(".", "-").replace("_", "-")
    return re.findall(r"\d{2}[-/]\d{2}[-/]\d{4}", text)


def extract_text_from_pdfs(pdf_files):
    content = ""
    for pdf in pdf_files:
        try:
            reader = PdfReader(pdf)
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    content += text + "\n"
        except Exception as e:
            st.warning(f"Erreur lecture PDF {pdf.name} : {e}")
    return content


def extract_text_from_images(image_files):
    content = ""
    for image_file in image_files:
        try:
            image = Image.open(image_file)
            processed = preprocess_image_for_ocr(image)

            text = pytesseract.image_to_string(
                processed,
                lang="fra+eng",
                config="--oem 3 --psm 6"
            )

            dates = extract_dates(text)
            if dates:
                text += "\nDates détectées : " + ", ".join(dates)

            if text.strip():
                content += f"\n[Texte extrait de {image_file.name}]\n{text}\n"

        except Exception as e:
            st.warning(f"Erreur lecture image {image_file.name} : {e}")

    return content


def build_vectorstore(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    chunks = splitter.split_text(text)

    if not chunks:
        return None, 0

    embeddings = OpenAIEmbeddings()
    temp_dir = tempfile.mkdtemp()

    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        collection_name="multimodal_rag_collection",
        persist_directory=temp_dir
    )
    return vectorstore, len(chunks)


def ask_question(question, retriever):
    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = PROMPT_TEMPLATE.format(
        context=context,
        question=question
    )

    response = llm.invoke(prompt)
    return response.content, docs


def init_session():
    if "retriever" not in st.session_state:
        st.session_state.retriever = None

    if "docs_ready" not in st.session_state:
        st.session_state.docs_ready = False

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hello! I'm your AI assistant. How can I help you today?"
            }
        ]

    if "sources" not in st.session_state:
        st.session_state.sources = []


def main():
    st.set_page_config(page_title="AI Assistant", page_icon="🤖", layout="wide")
    init_session()

    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 7rem;
            max-width: 1100px;
        }

        [data-testid="stHeader"] {
            background: rgba(0,0,0,0);
        }

        .app-header {
            padding: 0.3rem 0 1rem 0;
            border-bottom: 1px solid #e5e5e5;
            margin-bottom: 1rem;
        }

        .app-title {
            font-size: 1.8rem;
            font-weight: 700;
            margin-bottom: 0.1rem;
        }

        .app-subtitle {
            color: #666;
            font-size: 0.95rem;
        }

        [data-testid="stChatMessage"] {
            padding-top: 0.2rem;
            padding-bottom: 0.2rem;
        }

        [data-testid="stChatInput"] {
            position: fixed;
            bottom: 1rem;
            left: 55%;
            transform: translateX(-50%);
            width: min(900px, 92%);
            padding: 0.5rem 0.75rem;
            border-radius: 20px;
            box-shadow: 0 2px 12px rgba(0,0,0,0.08);
            z-index: 100;
        }

        section[data-testid="stSidebar"] {
            border-right: 1px solid #ececec;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="app-header">
            <div class="app-title"> Assistant</div>
            <div class="app-subtitle">Always here to help</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.subheader("Documents")

        pdf_files = st.file_uploader(
            "Ajouter des PDF",
            type=["pdf"],
            accept_multiple_files=True
        )

        image_files = st.file_uploader(
            "Ajouter des images",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True
        )

        if st.button("Traiter les documents", use_container_width=True):
            if not pdf_files and not image_files:
                st.warning("Veuillez ajouter au moins un PDF ou une image.")
            else:
                with st.spinner("Traitement en cours..."):
                    pdf_text = extract_text_from_pdfs(pdf_files) if pdf_files else ""
                    image_text = extract_text_from_images(image_files) if image_files else ""

                    full_text = pdf_text + "\n" + image_text

                    if not full_text.strip():
                        st.error("Aucun texte exploitable trouvé dans les fichiers.")
                    else:
                        vectorstore, nb_chunks = build_vectorstore(full_text)

                        if vectorstore is None:
                            st.error("Impossible de créer la base vectorielle.")
                        else:
                            st.session_state.retriever = vectorstore.as_retriever(
                                search_kwargs={"k": 8}
                            )
                            st.session_state.docs_ready = True
                            st.success(f"Documents prêts. {nb_chunks} chunks créés.")

        st.divider()

        if pdf_files:
            st.markdown("**PDF ajoutés :**")
            for pdf in pdf_files:
                st.write(f"- {pdf.name}")

        if image_files:
            st.markdown("**Images ajoutées :**")
            for img in image_files:
                st.write(f"- {img.name}")

        st.divider()

        if st.button("Effacer le chat", use_container_width=True):
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": "Hello! I'm your AI assistant. How can I help you today?"
                }
            ]
            st.session_state.sources = []
            st.rerun()

        if st.button("Nouveau chargement", use_container_width=True):
            st.session_state.retriever = None
            st.session_state.docs_ready = False
            st.session_state.sources = []
            st.rerun()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if st.session_state.sources:
        with st.expander("Contexte récupéré"):
            for i, doc in enumerate(st.session_state.sources, start=1):
                st.markdown(f"**Chunk {i}**")
                st.write(doc.page_content)

    user_question = st.chat_input("Type a message...")

    if user_question:
        st.session_state.messages.append(
            {"role": "user", "content": user_question}
        )

        if not st.session_state.docs_ready or st.session_state.retriever is None:
            answer = "Veuillez d'abord charger et traiter les documents dans la barre latérale."
            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )
            st.rerun()
        else:
            with st.spinner("Génération de la réponse..."):
                answer, docs = ask_question(
                    user_question,
                    st.session_state.retriever
                )

            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )
            st.session_state.sources = docs
            st.rerun()


if __name__ == "__main__":
    main()