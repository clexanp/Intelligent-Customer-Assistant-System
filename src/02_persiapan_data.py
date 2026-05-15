import re
import pandas as pd

JUMLAH_DATA = 120
LOKASI_INPUT = "data/dataset_assignment.csv"
LOKASI_OUTPUT = "data/knowledge_base_bersih.csv"


def bersihkan_teks(teks):
    if pd.isna(teks):
        return ""
    teks = str(teks)
    teks = teks.replace("\n", " ")
    teks = re.sub(r"\s+", " ", teks)
    return teks.strip()


def tentukan_kategori(teks):
    teks_kecil = teks.lower()

    if any(kata in teks_kecil for kata in ["refund", "return", "cancel", "policy", "terms", "privacy"]):
        return "policy"
    elif any(kata in teks_kecil for kata in ["error", "problem", "issue", "trouble", "not working", "failed"]):
        return "troubleshooting"
    elif any(kata in teks_kecil for kata in ["contact", "email", "phone", "support", "help"]):
        return "contact_information"
    else:
        return "faq"


def buat_tag(teks):
    teks_kecil = teks.lower()
    daftar_tag = []

    kamus_tag = {
        "akun": ["account", "login", "password", "sign in"],
        "pembayaran": ["payment", "invoice", "billing", "refund"],
        "teknis": ["error", "bug", "failed", "problem"],
        "kebijakan": ["policy", "terms", "privacy", "return"],
        "bantuan": ["help", "support", "contact", "email"],
    }

    for nama_tag, daftar_kata in kamus_tag.items():
        if any(kata in teks_kecil for kata in daftar_kata):
            daftar_tag.append(nama_tag)

    if not daftar_tag:
        daftar_tag.append("umum")

    return daftar_tag


def tentukan_tipe_informasi(kategori):
    if kategori == "policy":
        return "kebijakan_layanan"
    elif kategori == "troubleshooting":
        return "panduan_masalah"
    elif kategori == "contact_information":
        return "kontak_bantuan"
    else:
        return "tanya_jawab_umum"


df = pd.read_csv(LOKASI_INPUT)
df = df[["prompt", "response"]].copy()

df["pertanyaan"] = df["prompt"].apply(bersihkan_teks)
df["jawaban"] = df["response"].apply(bersihkan_teks)

df = df[(df["pertanyaan"] != "") & (df["jawaban"] != "")]
df = df.drop_duplicates(subset=["pertanyaan", "jawaban"])
df = df.head(JUMLAH_DATA).copy()

df["isi_dokumen"] = "Pertanyaan: " + df["pertanyaan"] + "\nJawaban: " + df["jawaban"]
df["kategori"] = df["isi_dokumen"].apply(tentukan_kategori)
df["tag"] = df["isi_dokumen"].apply(buat_tag)
df["tipe_informasi"] = df["kategori"].apply(tentukan_tipe_informasi)
df["sumber"] = "dataset_assignment"

kolom_final = [
    "pertanyaan",
    "jawaban",
    "isi_dokumen",
    "kategori",
    "tag",
    "tipe_informasi",
    "sumber",
]

df[kolom_final].to_csv(LOKASI_OUTPUT, index=False)

print("Knowledge base berhasil dibuat.")
print("Jumlah data:", len(df))
print("Lokasi output:", LOKASI_OUTPUT)
print("\nDistribusi kategori:")
print(df["kategori"].value_counts())
