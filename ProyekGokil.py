import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# 1. Memuat Dataset
df = pd.read_csv('smartphones_clean.csv')

# Memilih kolom kriteria yang akan digunakan
kriteria = ['price', 'rating', 'processor_speed', 'ram_gb', 'storage_gb', 'camera_mp', 'battery_mah']
df_spk = df[['model'] + kriteria].copy()

nama_kriteria = [
            "Harga",
            "Rating",
            "Prosesor",
            "RAM",
            "Storage",
            "Kamera",
            "Baterai"
        ]

# Data Preprocessing: Mengisi nilai kosong (NaN) dengan median dan mengatasi nilai 0/negatif
for col in kriteria:
    df_spk[col] = df_spk[col].fillna(df_spk[col].median())
    df_spk[col] = df_spk[col].apply(lambda x: 1e-5 if x <= 0 else x)

# 2. Persiapan Matriks Keputusan dan Parameter (Sesuai Modul Praktikum)
alternatif = df_spk['model'].values
# Matriks data (diubah ke numpy array agar bisa dihitung dengan np.prod seperti di modul)
data = df_spk[kriteria].values 

# setup halaman & dataframe alternatif
st.set_page_config(page_title="SPK Smartphone dengan WP", layout="wide")

pg1_smartphone = df_spk['model']
pg1_harga = df_spk['price']
pg1_rating = df_spk['rating']
pg1_prosesor = df_spk['processor_speed']
pg1_ram = df_spk['ram_gb']
pg1_storage = df_spk['storage_gb']
pg1_kamera = df_spk['camera_mp']
pg1_baterai = df_spk['battery_mah']

df = pd.DataFrame({
    "Smartphone": pg1_smartphone,
    "Harga (Juta Rp)": pg1_harga,
    "Rating": pg1_rating,    
    "Prosessor (GHz)": pg1_prosesor,
    "RAM (GB)": pg1_ram,
    "Storage (GB)": pg1_storage,
    "Kamera (MP)": pg1_kamera,
    "Baterai (MAh)": pg1_baterai
})

# Sidebar pengaturan bobot
st.sidebar.title("Pengaturan")
pilihan = st.sidebar.selectbox(
    "Pilih Halaman : ", 
    ("Page 1 - Data Alternatif", "Page 2 - WP", "Page 3 - Perhitungan WP", "Page 4 - Visualisasi")
)

st.sidebar.subheader("Slider Bobot")
st.sidebar.caption("Bobot menunjukkan tingkat kepentingan setiap kriteria dalam proses pengambilan keputusan.")
bobot_harga = st.sidebar.slider("Harga (Juta Rp)", min_value=1.0, max_value=5.0, value=4.0, step=0.1)
bobot_rating = st.sidebar.slider("Rating", min_value=1.0, max_value=5.0, value=3.0, step=0.1)
bobot_prosesor = st.sidebar.slider("Prosessor (GHz)", min_value=1.0, max_value=5.0, value=3.0, step=0.1)
bobot_ram = st.sidebar.slider("RAM (GB)", min_value=1.0, max_value=5.0, value=3.0, step=0.1)
bobot_storage = st.sidebar.slider("Storage (GB)", min_value=1.0, max_value=5.0, value=4.0, step=0.1)
bobot_kamera = st.sidebar.slider("Kamera (MP)", min_value=1.0, max_value=5.0, value=2.0, step=0.1)
bobot_baterai = st.sidebar.slider("Baterai (mAh)", min_value=1.0, max_value=5.0, value=1.0, step=0.1)
# hitung bobot ternormalisasi
total_bobot = bobot_harga + bobot_rating + bobot_prosesor + bobot_ram + bobot_storage + bobot_kamera + bobot_baterai
w_harga = bobot_harga / total_bobot
w_rating = bobot_rating / total_bobot
w_prosesor = bobot_prosesor / total_bobot
w_ram = bobot_ram / total_bobot
w_storage = bobot_storage / total_bobot
w_kamera = bobot_kamera / total_bobot
w_baterai = bobot_baterai / total_bobot

# Menghitung WP anjay
# Bobot Preferensi Awal
bobot_awal = np.array([bobot_harga, bobot_rating, bobot_prosesor, bobot_ram, bobot_storage, bobot_kamera, bobot_baterai])

# Penentuan Atribut Cost/Benefit (-1 = Cost, 1 = Benefit)
k = np.array([-1, 1, 1, 1, 1, 1,1])

# 3. Normalisasi Nilai Bobot
norm_bobot = np.array([w_harga, w_rating, w_prosesor, w_ram, w_storage, w_kamera, w_baterai])

# 4. Menghitung Nilai Preferensi (Vektor S)
s = np.prod(data ** (k * norm_bobot), axis=1)

# 5. Normalisasi Vektor S (Vektor V)
v = s / np.sum(s)

# 6. Menggabungkan Hasil ke DataFrame dan Perangkingan
hasil_df = pd.DataFrame({
    'Model': alternatif,
    'Vektor S': s,
    'Vektor V': v
    })
        
sort_ranking = hasil_df.sort_values(by='Vektor V', ascending=False).reset_index(drop=True)

pemenang_wp = sort_ranking.iloc[0]["Model"]


st.sidebar.subheader("Bobot Ternormalisasi")
st.sidebar.caption("Normalisasi bobot dilakukan agar total seluruh bobot bernilai 1.")
st.sidebar.caption(f"Harga: {w_harga:.4f}")
st.sidebar.caption(f"Rating: {w_rating:.4f}")
st.sidebar.caption(f"Prosesor: {w_prosesor:.4f}")
st.sidebar.caption(f"RAM: {w_ram:.4f}")
st.sidebar.caption(f"Storage: {w_storage:.4f}")
st.sidebar.caption(f"Kamera: {w_kamera:.4f}")
st.sidebar.caption(f"Baterai: {w_baterai:.4f}")
# navigasi halaman
match pilihan:
    #halaman 1
    case "Page 1 - Data Alternatif":
        st.header("Data Alternatif Smartphone")
        st.write("Sistem pendukung keputusan ini membantu memilih smartphone terbaik berdasarkan 7 kriteria utama menggunakan metode WP.")
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.subheader("Keterangan Kriteria")
        col1, col2, col3, col4 = st.columns(4)
        col5, col6, col7 = st.columns(3)
        
        with col1:
            st.error("**Harga (Juta Rp)**\n\nTipe: Cost\n\n*Semakin murah semakin baik*")
        with col2:
            st.success("**Rating (Nilai)**\n\nTipe: Benefit\n\n*Semakin besar semakin baik*")
        with col3:
            st.success("**Prosesor (MHz)**\n\nTipe: Benefit\n\n*Semakin besar semakin baik*")
        with col4:
            st.success("**RAM (GB)**\n\nTipe: Benefit\n\n*Semakin besar semakin baik*")
        with col5:
            st.success("**Storage (GB)**\n\nTipe: Benefit\n\n*Semakin besar semakin baik*")
        with col6:
            st.success("**Kamera (MP)**\n\nTipe: Benefit\n\n*Semakin besar semakin baik*")
        with col7:
            st.success("**Baterai (mAh)**\n\nTipe: Benefit\n\n*Semakin besar semakin baik*")
        
        st.caption("Sistem Pendukung Keputusan Pemilihan Smartphone dengan Metode Weighted Product")

    #halaman 2
    case "Page 2 - WP":
        st.header("Penjelasan Metode Weighted Product")

        st.subheader("Pengertian Metode Weighted Product")

        st.markdown("""
        Metode Weighted Product (WP) merupakan metode Sistem Pendukung Keputusan (SPK)
        yang digunakan untuk menentukan alternatif terbaik berdasarkan beberapa kriteria.
        Metode ini bekerja dengan cara mengalikan setiap nilai kriteria yang telah dipangkatkan
        dengan bobot masing-masing kriteria.
        """)

        st.subheader("Tujuan Metode")

        st.markdown("""
        Pada sistem ini, metode Weighted Product digunakan untuk memberikan rekomendasi
        smartphone terbaik berdasarkan beberapa kriteria:
        - Harga
        - Rating
        - Processor
        - RAM
        - Storage
        - Kamera
        - Baterai
        """)

        st.subheader("Jenis Kriteria")

        st.markdown("""
        Dalam metode WP terdapat dua jenis atribut:
        - Benefit → semakin besar nilainya semakin baik
        - Cost → semakin kecil nilainya semakin baik
        """)

        jenis_df = pd.DataFrame({
            "Kriteria": nama_kriteria,
            "Jenis": ["Cost", "Benefit", "Benefit", "Benefit", "Benefit", "Benefit", "Benefit"]
        })

        st.dataframe(jenis_df, use_container_width=True, hide_index=True)

        st.subheader("Normalisasi Bobot")

        st.markdown(r"""
        Bobot setiap kriteria dinormalisasi agar total seluruh bobot bernilai 1.

        $$
        w_j = \frac{w_j}{\sum w_j}
        $$

        Keterangan:
        - $w_j$ = bobot normalisasi
        - $\sum w_j$ = total seluruh bobot
        """)

        st.subheader("Perhitungan Vektor S")

        st.markdown(r"""
        Vektor S digunakan untuk menghitung nilai preferensi setiap alternatif.

        $$
        S_i = \prod_{j=1}^{n} x_{ij}^{w_j}
        $$

        Keterangan:
        - $S_i$ = nilai preferensi alternatif ke-i
        - $x_{ij}$ = nilai alternatif terhadap kriteria
        - $w_j$ = bobot kriteria
        """)

        st.info("""
        Pada atribut Cost seperti harga, bobot akan bernilai negatif
        agar nilai yang lebih kecil menjadi lebih baik.
        """)

        st.subheader("Perhitungan Vektor V")

        st.markdown(r"""
        Vektor V digunakan untuk normalisasi hasil perhitungan Vektor S.

        $$
        V_i = \frac{S_i}{\sum S_i}
        $$

        Keterangan:
        - $V_i$ = nilai akhir alternatif
        - Nilai terbesar menjadi alternatif terbaik
        """)

        st.subheader("Langkah Metode Weighted Product")

        langkah_wp = pd.DataFrame({
            "Tahapan": [
                "1. Menentukan kriteria",
                "2. Menentukan bobot kriteria",
                "3. Melakukan normalisasi bobot",
                "4. Menentukan atribut cost dan benefit",
                "5. Menghitung Vektor S",
                "6. Menghitung Vektor V",
                "7. Melakukan perangkingan"
            ]
        })

        st.dataframe(langkah_wp, use_container_width=True, hide_index=True)
        
        st.caption("Sistem Pendukung Keputusan Pemilihan Smartphone dengan Metode Weighted Product")
    
    #halaman 3
    case "Page 3 - Perhitungan WP":
        st.header("Hasil Perhitungan WP")
        
        st.subheader("Matriks Keputusan Awal (Top 5 baris)")
        st.dataframe(pd.DataFrame(data[:5], index=alternatif[:5], columns=nama_kriteria))
        
        bobot_df = pd.DataFrame({
            "Kriteria" : nama_kriteria,
            "Bobot Awal" : bobot_awal,
            "Bobot Ternormalisasi" : norm_bobot
        })
        st.subheader("Tabel Bobot Kriteria")
        st.dataframe(bobot_df, use_container_width=True, hide_index=True)       

        st.subheader("\nHasil Ranking (Top 10)")
        st.dataframe(sort_ranking.head(10), use_container_width=True, hide_index=True)
        
        skor_wp = sort_ranking.iloc[0]["Vektor V"]
        st.success(f" **Kesimpulan WP:** ✧｡◝(ᵔᗜᵔ)◜✧*｡\n\nBerdasarkan metode WP, rekomendasi utama adalah **{pemenang_wp}** dengan nilai **{skor_wp:.4f}**")
        
        st.caption("Sistem Pendukung Keputusan Pemilihan Smartphone dengan Metode Weighted Product")
    
        
    case "Page 4 - Visualisasi":

        st.header("Visualisasi Data")
        st.subheader("Grafik Bobot Kriteria")
        
        st.subheader("1. Proporsi Bobot Kriteria (Pie Chart)")
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        
        warna1 = [
            '#EF4444',  # Harga
            '#F97316',  # Rating
            '#EAB308',  # Prosesor
            '#22C55E',  # RAM
            '#06B6D4',  # Storage
            '#3B82F6',  # Kamera
            '#8B5CF6'   # Baterai
        ]
        
        # Menggunakan autopct untuk menampilkan persentase
        ax1.pie(norm_bobot, labels=nama_kriteria, autopct='%1.1f%%', startangle=140, colors=warna1)
        ax1.axis('equal') # Memastikan pie chart berbentuk lingkaran sempurna
        ax1.set_title("Persentase Bobot Setiap Kriteria")
        st.pyplot(fig1)
        
        st.subheader("Grafik Ranking Smartphone")
        
        top10 = sort_ranking.head(10)
        
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        
        ax2.bar(top10['Model'], top10['Vektor V'])
        
        ax2.set_xlabel("Model Smartphone")
        ax2.set_ylabel("Nilai Vektor V")
        ax2.set_title("Top 10 Ranking Smartphone Metode WP")
        
        plt.xticks(rotation=90)
        
        st.pyplot(fig2)
        
        data_pemenang = df_spk[df_spk['model'] == pemenang_wp].iloc[0]
        
        st.subheader(f"Profil Kriteria Pemenang: {pemenang_wp}")

        kriteria_pemenang = [
            data_pemenang['price'] ** (-w_harga),      # cost
        data_pemenang['rating'] ** w_rating,
        data_pemenang['processor_speed'] ** w_prosesor,
        data_pemenang['ram_gb'] ** w_ram,
        data_pemenang['storage_gb'] ** w_storage,
        data_pemenang['camera_mp'] ** w_kamera,
        data_pemenang['battery_mah'] ** w_baterai
        ]
        
        fig, ax = plt.subplots(figsize=(10,5))

        ax.barh(nama_kriteria, kriteria_pemenang)

        ax.set_title(f"Nilai Kriteria Smartphone Pemenang\n{pemenang_wp}")
        ax.set_xlabel("Nilai")

        for i, v in enumerate(kriteria_pemenang):
            ax.text(v, i, str(v), va='center')

        st.pyplot(fig)
        
        st.caption("Sistem Pendukung Keputusan Pemilihan Smartphone dengan Metode Weighted Product")