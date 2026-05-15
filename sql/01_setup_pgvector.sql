CREATE EXTENSION IF NOT EXISTS vector;

DROP TABLE IF EXISTS knowledge_base;

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

CREATE INDEX IF NOT EXISTS idx_kb_kategori
ON knowledge_base(kategori);

CREATE INDEX IF NOT EXISTS idx_kb_tipe
ON knowledge_base(tipe_informasi);

CREATE INDEX IF NOT EXISTS idx_kb_embedding_hnsw
ON knowledge_base
USING hnsw (embedding vector_cosine_ops);
