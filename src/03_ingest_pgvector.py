import ast
import pandas as pd
import psycopg
from sentence_transformers import SentenceTransformer
from pgvector.psycopg import register_vector

from config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD, EMBEDDING_MODEL

LOKASI_KB = "data/knowledge_base_bersih.csv"


def buka_koneksi():
    conn = psycopg.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        autocommit=True,
    )
    register_vector(conn)
    return conn


def siapkan_tabel(conn):
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute("DROP TABLE IF EXISTS knowledge_base;")
        cur.execute(
            """
            CREATE TABLE knowledge_base (
                id SERIAL PRIMARY KEY,
                pertanyaan TEXT NOT NULL,
                jawaban TEXT NOT NULL,
                isi_dokumen TEXT NOT NULL,
                kategori VARCHAR(100),
                tag TEXT[],
                tipe_informasi VARCHAR(100),
                sumber VARCHAR(100),
                embedding vector(384),
                dibuat_pada TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        cur.execute("CREATE INDEX idx_kb_kategori ON knowledge_base(kategori);")
        cur.execute("CREATE INDEX idx_kb_tipe ON knowledge_base(tipe_informasi);")
        cur.execute(
            """
            CREATE INDEX idx_kb_embedding_hnsw
            ON knowledge_base
            USING hnsw (embedding vector_cosine_ops);
            """
        )


def simpan_data(conn, df):
    model_embedding = SentenceTransformer(EMBEDDING_MODEL)

    daftar_teks = df["isi_dokumen"].tolist()
    daftar_embedding = model_embedding.encode(
        daftar_teks,
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    with conn.cursor() as cur:
        for baris, embedding in zip(df.to_dict("records"), daftar_embedding):
            cur.execute(
                """
                INSERT INTO knowledge_base
                (pertanyaan, jawaban, isi_dokumen, kategori, tag, tipe_informasi, sumber, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    baris["pertanyaan"],
                    baris["jawaban"],
                    baris["isi_dokumen"],
                    baris["kategori"],
                    baris["tag"],
                    baris["tipe_informasi"],
                    baris["sumber"],
                    embedding.tolist(),
                ),
            )


def main():
    df = pd.read_csv(LOKASI_KB)
    df["tag"] = df["tag"].apply(ast.literal_eval)

    conn = buka_koneksi()
    siapkan_tabel(conn)
    simpan_data(conn, df)
    conn.close()

    print("Data knowledge base dan embedding berhasil masuk ke PostgreSQL + PgVector.")
    print("Jumlah data:", len(df))


if __name__ == "__main__":
    main()
