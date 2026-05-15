import sys
sys.path.append("src")

import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

from config import GROQ_API_KEY, GROQ_MODEL
from mesin_pencarian import hybrid_search

st.set_page_config(
    page_title="Intelligent Customer Assistant System",
    page_icon="💬",
    layout="centered",
)

st.title("💬 Intelligent Customer Assistant System")

if "riwayat_chat" not in st.session_state:
    st.session_state.riwayat_chat = []

with st.sidebar:
    st.header("Pengaturan Retrieval")
    kategori = st.selectbox(
        "Filter kategori",
        ["semua", "faq", "policy", "troubleshooting", "contact_information"],
    )
    jumlah_context = st.slider("Jumlah context", 3, 8, 5)

    st.markdown("---")
    st.write("Sistem ini mengambil context dari PostgreSQL + PgVector, lalu Groq membuat jawaban akhir.")

for chat in st.session_state.riwayat_chat:
    with st.chat_message(chat["role"]):
        st.markdown(chat["content"])

pertanyaan_user = st.chat_input("Tulis pertanyaan anda di sini...")

template_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Kamu adalah customer assistant yang ramah, jelas, dan tidak mengarang jawaban.
            Jawab hanya berdasarkan context knowledge base yang diberikan.
            Jika informasi tidak ditemukan, bilang bahwa data belum tersedia dan arahkan user untuk menghubungi support.
            Gunakan bahasa Indonesia yang natural dan mudah dipahami.
            """,
        ),
        (
            "human",
            """
            Pertanyaan customer:
            {pertanyaan}

            Context dari knowledge base:
            {context}

            Buat jawaban final yang singkat, membantu, dan sopan.
            """,
        ),
    ]
)

if pertanyaan_user:
    st.session_state.riwayat_chat.append(
        {"role": "user", "content": pertanyaan_user}
    )

    with st.chat_message("user"):
        st.markdown(pertanyaan_user)

    with st.chat_message("assistant"):
        with st.spinner("Sedang mencari jawaban paling relevan..."):
            hasil_retrieval = hybrid_search(
                pertanyaan_user,
                kategori=kategori,
                batas=jumlah_context,
            )

            if not hasil_retrieval:
                jawaban_final = (
                    "Maaf, aku belum menemukan informasi yang cukup relevan di knowledge base. "
                    "Silakan hubungi tim support agar pertanyaanmu bisa dibantu lebih lanjut."
                )
            else:
                context = "\n\n".join(
                    [
                        f"[Context {idx}]\n"
                        f"Kategori: {item['kategori']}\n"
                        f"Skor relevansi: {item['skor_relevansi']}\n"
                        f"Pertanyaan referensi: {item['pertanyaan']}\n"
                        f"Jawaban referensi: {item['jawaban']}"
                        for idx, item in enumerate(hasil_retrieval, start=1)
                    ]
                )

                llm = ChatGroq(
                    groq_api_key=GROQ_API_KEY,
                    model_name=GROQ_MODEL,
                    temperature=0.2,
                )

                chain = template_prompt | llm
                response = chain.invoke(
                    {
                        "pertanyaan": pertanyaan_user,
                        "context": context,
                    }
                )

                jawaban_final = response.content

            st.markdown(jawaban_final)

            with st.expander("Lihat context retrieval"):
                for item in hasil_retrieval:
                    st.write(
                        {
                            "skor_relevansi": item["skor_relevansi"],
                            "kategori": item["kategori"],
                            "pertanyaan": item["pertanyaan"],
                        }
                    )

    st.session_state.riwayat_chat.append(
        {"role": "assistant", "content": jawaban_final}
    )
