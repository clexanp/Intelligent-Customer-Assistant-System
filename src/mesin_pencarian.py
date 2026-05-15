import time
import psycopg
from pgvector.psycopg import register_vector
from sentence_transformers import SentenceTransformer

from config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD, EMBEDDING_MODEL

model_embedding = SentenceTransformer(EMBEDDING_MODEL)


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


def cari_semantik(query, kategori=None, batas=5, threshold=0.35):
    query_bersih = query.strip()
    embedding_query = model_embedding.encode(
        query_bersih,
        normalize_embeddings=True,
    ).tolist()

    kondisi_filter = ""
    parameter = [embedding_query, embedding_query]

    if kategori and kategori != "semua":
        kondisi_filter = "WHERE kategori = %s"
        parameter.append(kategori)

    parameter.append(batas)

    sql = f"""
        SELECT
            id,
            pertanyaan,
            jawaban,
            kategori,
            tag,
            tipe_informasi,
            1 - (embedding <=> %s::vector) AS skor_relevansi
        FROM knowledge_base
        {kondisi_filter}
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """

    waktu_mulai = time.time()

    conn = buka_koneksi()
    with conn.cursor() as cur:
        cur.execute(sql, parameter)
        hasil = cur.fetchall()
    conn.close()

    durasi = round(time.time() - waktu_mulai, 4)

    daftar_hasil = []
    for item in hasil:
        skor = float(item[6])
        if skor >= threshold:
            daftar_hasil.append(
                {
                    "id": item[0],
                    "pertanyaan": item[1],
                    "jawaban": item[2],
                    "kategori": item[3],
                    "tag": item[4],
                    "tipe_informasi": item[5],
                    "skor_relevansi": round(skor, 4),
                }
            )

    print(f"Query: {query}")
    print(f"Durasi pencarian: {durasi} detik")
    print(f"Jumlah hasil relevan: {len(daftar_hasil)}")

    return daftar_hasil


def cari_exact_match(query, batas=3):
    conn = buka_koneksi()
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, pertanyaan, jawaban, kategori
            FROM knowledge_base
            WHERE pertanyaan ILIKE %s OR jawaban ILIKE %s
            LIMIT %s;
            """,
            (f"%{query}%", f"%{query}%", batas),
        )
        hasil = cur.fetchall()
    conn.close()

    return [
        {
            "id": item[0],
            "pertanyaan": item[1],
            "jawaban": item[2],
            "kategori": item[3],
            "skor_relevansi": 1.0,
        }
        for item in hasil
    ]


def hybrid_search(query, kategori=None, batas=5):
    hasil_exact = cari_exact_match(query, batas=2)
    hasil_semantik = cari_semantik(query, kategori=kategori, batas=batas)

    id_terpakai = set()
    hasil_gabungan = []

    for item in hasil_exact + hasil_semantik:
        if item["id"] not in id_terpakai:
            hasil_gabungan.append(item)
            id_terpakai.add(item["id"])

    return hasil_gabungan[:batas]


if __name__ == "__main__":
    hasil = hybrid_search("how can I contact support?", kategori="semua")
    for nomor, item in enumerate(hasil, start=1):
        print("\nHasil", nomor)
        print("Skor:", item["skor_relevansi"])
        print("Kategori:", item["kategori"])
        print("Pertanyaan:", item["pertanyaan"])
        print("Jawaban:", item["jawaban"][:300])
