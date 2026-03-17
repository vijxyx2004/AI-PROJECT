import streamlit as st
import os

from rag import load_pdf, search_endee, split_text, store_in_endee

st.title("📄 AI PDF Chatbot using Endee")

endee_url = st.text_input("Endee URL", value=os.getenv("ENDEE_URL", "http://localhost:8080"))
index_name = st.text_input("Endee index name", value=os.getenv("ENDEE_INDEX", "docs"))
top_k = st.slider("Top K", 1, 10, 3)

st.divider()
uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if "pdf_ingested" not in st.session_state:
    st.session_state.pdf_ingested = False
    st.session_state.last_pdf_name = ""

if uploaded_file:
    if st.button("Ingest PDF into Endee", type="primary"):
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.read())

        text = load_pdf("temp.pdf")
        chunks = split_text(text, chunk_size=1200, overlap=200)

        if not text.strip() or not chunks:
            st.warning(
                "No extractable text found in this PDF. If it's a scanned PDF (image), "
                "you need OCR to extract text before RAG will work."
            )
        else:
            try:
                inserted = store_in_endee(
                    chunks,
                    endee_url=endee_url,
                    index_name=index_name,
                    source_name=uploaded_file.name,
                )
                if inserted <= 0:
                    st.warning("No chunks were stored in Endee (nothing to index).")
                else:
                    st.session_state.pdf_ingested = True
                    st.session_state.last_pdf_name = uploaded_file.name
                    st.success(f"PDF stored in Endee! ({inserted} chunks)")
            except ConnectionError as e:
                st.error(str(e))
                st.info(
                    "Start Endee (Docker) so `http://localhost:8080/api/v1/health` returns 200, then try again."
                )
            except Exception as e:
                st.error(f"Failed to store vectors in Endee: {e}")
else:
    st.info("Upload a PDF to ingest it into Endee.")

st.divider()
st.subheader("Ask a question")

if st.session_state.pdf_ingested:
    st.caption(f"Using Endee index `{index_name}` (last ingested: {st.session_state.last_pdf_name})")
else:
    st.caption(f"Using Endee index `{index_name}`. Ingest a PDF first for best results.")

query = st.text_input("Question")
ask = st.button("Answer", disabled=not query)

if ask and query:
    try:
        results = search_endee(query, endee_url=endee_url, index_name=index_name, top_k=top_k)
        st.write("### Retrieved passages:")
        for r in results:
            st.write(r)
    except ConnectionError as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"Search failed: {e}")