import streamlit as st
import pandas as pd
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import time
import itertools

st.set_page_config(
    page_title="SPK Pemilihan Smartphone - Fuzzy Mamdani",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================
# INISIALISASI SESSION STATE
# ==========================================
# Parameter batas 'Sedang' / 'Tengah' untuk masing-masing kriteria
if "param_harga" not in st.session_state:
    st.session_state.param_harga = 50.0
if "param_ram" not in st.session_state:
    st.session_state.param_ram = 50.0
if "param_storage" not in st.session_state:
    st.session_state.param_storage = 50.0
if "param_baterai" not in st.session_state:
    st.session_state.param_baterai = 50.0

if "defuzz_method" not in st.session_state:
    st.session_state.defuzz_method = "Centroid"

if "top_n" not in st.session_state:
    st.session_state.top_n = 10

if "df_data" not in st.session_state:
    st.session_state.df_data = None


# ==========================================
# FUNGSI CACHE UNTUK LOAD DATA
# ==========================================
@st.cache_data
def load_data(uploaded_file=None):
    try:
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_csv("smartphones_clean.csv")

        # Pre-processing: Normalisasi data menjadi persentil (0-100)
        # Agar mudah diproses di mesin fuzzy yang rentangnya 0-100
        df_spk = df.copy()
        
        # Benefit: Makin besar nilai asli, persentil makin mendekati 100
        df_spk['RAM_pct'] = df_spk['ram_gb'].rank(pct=True) * 100
        df_spk['Storage_pct'] = df_spk['storage_gb'].rank(pct=True) * 100
        df_spk['Baterai_pct'] = df_spk['battery_mah'].rank(pct=True) * 100
        
        # Cost: Makin MURA (kecil) harga asli, persentil makin mendekati 100 (Keterjangkauan Tinggi)
        df_spk['Harga_pct'] = (1 - df_spk['price'].rank(pct=True)) * 100
        
        return df_spk

    except FileNotFoundError:
        return None
    except Exception as e:
        st.warning(f"Terjadi error: {e}")
        return None


# ==========================================
# FUNGSI PEMBUAT SISTEM FUZZY
# ==========================================
def fuzzy_system(p_harga, p_ram, p_stor, p_bat, defuzz_method):

    # 1. Deklarasi Variabel Antecedent (0-100 Persentil)
    x_range = np.arange(0, 101, 1)
    
    harga = ctrl.Antecedent(x_range, "harga")
    ram = ctrl.Antecedent(x_range, "ram")
    storage = ctrl.Antecedent(x_range, "storage")
    baterai = ctrl.Antecedent(x_range, "baterai")

    score = ctrl.Consequent(
        x_range, "score", defuzzify_method=defuzz_method.lower()
    )

    # 2. Fungsi Keanggotaan (Otomatis menyesuaikan parameter slider)
    def create_mf(antecedent, param):
        b, a = max(0, param - 25), min(100, param + 25)
        antecedent['rendah'] = fuzz.trimf(x_range, [0, 0, param])
        antecedent['sedang'] = fuzz.trimf(x_range, [b, param, a])
        antecedent['tinggi'] = fuzz.trimf(x_range, [param, 100, 100])

    create_mf(harga, p_harga)
    create_mf(ram, p_ram)
    create_mf(storage, p_stor)
    create_mf(baterai, p_bat)

    # Output Rekomendasi
    score["buruk"] = fuzz.trapmf(score.universe, [0, 0, 30, 45])
    score["cukup"] = fuzz.trimf(score.universe, [35, 50, 65])
    score["baik"] = fuzz.trimf(score.universe, [55, 75, 90])
    score["sangat_baik"] = fuzz.trapmf(score.universe, [80, 90, 100, 100])

    # 3. Rule Generation (81 Aturan Otomatis Menggunakan Itertools)
    rules = []
    terms = ['rendah', 'sedang', 'tinggi']
    score_map = {'rendah': 1, 'sedang': 2, 'tinggi': 3}
    
    for h, r, s, b in itertools.product(terms, repeat=4):
        total_score = score_map[h] + score_map[r] + score_map[s] + score_map[b]
        
        # Logika pembagian kuadran output
        if total_score <= 5:
            out = 'buruk'
        elif total_score <= 8:
            out = 'cukup'
        elif total_score <= 10:
            out = 'baik'
        else:
            out = 'sangat_baik'
            
        rule = ctrl.Rule(
            harga[h] & ram[r] & storage[s] & baterai[b],
            score[out]
        )
        rules.append(rule)

    hp_ctrl = ctrl.ControlSystem(rules)
    hp_sim = ctrl.ControlSystemSimulation(hp_ctrl)
    
    return hp_sim, harga, ram, storage, baterai, score


# ==========================================
# FUNGSI HELPER
# ==========================================
def tingkat_rekomendasi(crisp_val):
    if crisp_val <= 35:
        return "Buruk"
    elif crisp_val <= 60:
        return "Cukup"
    elif crisp_val <= 85:
        return "Baik"
    else:
        return "Sangat Baik"


# ==========================================
# SIDEBAR NAVIGATION
# ==========================================
st.sidebar.title("🧭 Navigasi SPK")
menu = st.sidebar.selectbox(
    "Pilih Halaman:",
    [
        "📊 Dataset",
        "⚙️ Konfigurasi Fuzzy",
        "🏆 Hitung & Peringkat SPK",
        "👥 Tentang Program",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown(''':shimmer[**Dikembangkan oleh:**] :rainbow[**Alex & Rifty**]''', width="auto")

# ==========================================
# HALAMAN 1: DATASET
# ==========================================
if menu == "📊 Dataset":
    st.title("📊 Dataset Smartphone")

    st.session_state.df_data = load_data()
    df = st.session_state.df_data

    if df is not None:
        st.subheader("Dataset (Data Mentah & Kolom Normalisasi)")
        st.dataframe(df.head(100), use_container_width=True) # Tampilkan 100 baris agar tidak lag

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Statistik Deskriptif")
            st.write(df[['price', 'ram_gb', 'storage_gb', 'battery_mah']].describe())

        with col2:
            st.subheader("Distribusi Harga")
            df_mahal = df.sort_values(by='price', ascending=False).head(10)
            st.bar_chart(data=df_mahal, x='model', y='price')
    else:
        st.error("Dataset tidak ditemukan. Silakan upload file CSV.")

# ==========================================
# HALAMAN 2: KONFIGURASI FUZZY
# ==========================================
elif menu == "⚙️ Konfigurasi Fuzzy":
    st.title("⚙️ Konfigurasi Parameter Fuzzy Mamdani")

    st.markdown("### Pengaturan Batas Variabel ('Sedang')")
    st.caption("Geser slider untuk menentukan batas titik tengah persentil untuk setiap kriteria.")
    
    col_w1, col_w2 = st.columns(2)

    with col_w1:
        st.session_state.param_harga = st.slider(
            "Batas Keterjangkauan Harga", 30.0, 70.0, st.session_state.param_harga, step=1.0
        )
        st.session_state.param_ram = st.slider(
            "Batas Kapasitas RAM", 30.0, 70.0, st.session_state.param_ram, step=1.0
        )

    with col_w2:
        st.session_state.param_storage = st.slider(
            "Batas Kapasitas Storage", 30.0, 70.0, st.session_state.param_storage, step=1.0
        )
        st.session_state.param_baterai = st.slider(
            "Batas Kapasitas Baterai", 30.0, 70.0, st.session_state.param_baterai, step=1.0
        )

    st.markdown("### Pengaturan Sistem")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        opsi_defuzz = ["Centroid", "Bisector", "MOM"]
        default_idx = opsi_defuzz.index(st.session_state.defuzz_method) if st.session_state.defuzz_method in opsi_defuzz else 0
        st.session_state.defuzz_method = st.selectbox(
            "Metode Defuzzifikasi", options=opsi_defuzz, index=default_idx
        )
    with col_s2:
        st.session_state.top_n = st.number_input(
            "Jumlah Top-N Hasil Peringkat", min_value=5, max_value=50, value=st.session_state.top_n, step=5
        )

    st.markdown("---")

    # Generate Simulasi Sementar untuk Grafik
    sim, harga, ram, storage, baterai, score = fuzzy_system(
        st.session_state.param_harga, st.session_state.param_ram, 
        st.session_state.param_storage, st.session_state.param_baterai, 
        st.session_state.defuzz_method
    )

    st.subheader("📈 Kurva Fungsi Keanggotaan")
    viz_option = st.selectbox(
        "Pilih Variabel untuk Divisualisasikan:",
        ["Keterjangkauan Harga", "Kapasitas RAM", "Kapasitas Storage", "Kapasitas Baterai", "Output Skor Rekomendasi"],
    )

    plt.close("all")

    if viz_option == "Keterjangkauan Harga":
        harga.view()
        plt.title("Fungsi Keanggotaan: Harga")
    elif viz_option == "Kapasitas RAM":
        ram.view()
        plt.title("Fungsi Keanggotaan: RAM")
    elif viz_option == "Kapasitas Storage":
        storage.view()
        plt.title("Fungsi Keanggotaan: Storage")
    elif viz_option == "Kapasitas Baterai":
        baterai.view()
        plt.title("Fungsi Keanggotaan: Baterai")
    elif viz_option == "Output Skor Rekomendasi":
        score.view()
        plt.title("Fungsi Keanggotaan: Skor Kelayakan (Output)")

    fig = plt.gcf()
    fig.set_size_inches(10, 4)
    st.pyplot(fig)

# ==========================================
# HALAMAN 3: HITUNG & PERINGKAT SPK
# ==========================================
elif menu == "🏆 Hitung & Peringkat SPK":
    st.title("🏆 Perhitungan SPK Smartphone")

    df = st.session_state.df_data

    if df is None:
        st.warning("Dataset belum diload! Silakan ke halaman 'Dataset' dan Upload file CSV terlebih dahulu.")
    else:
        if st.button("🚀 Hitung Peringkat Fuzzy", use_container_width=True, type="primary"):
            t_mulai = time.time()

            data_rows = df.to_dict("records")

            # Bikin Master Control dari session state
            master_sim, _, _, _, _, _ = fuzzy_system(
                st.session_state.param_harga,
                st.session_state.param_ram,
                st.session_state.param_storage,
                st.session_state.param_baterai,
                st.session_state.defuzz_method
            )
            master_ctrl = master_sim.ctrl

            skor_hasil = []
            total_data = len(data_rows)

            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, row in enumerate(data_rows):
                sim = ctrl.ControlSystemSimulation(master_ctrl)

                # Masukkan nilai yang sudah dinormalisasi persentil (0-100)
                sim.input["harga"] = row["Harga_pct"]
                sim.input["ram"] = row["RAM_pct"]
                sim.input["storage"] = row["Storage_pct"]
                sim.input["baterai"] = row["Baterai_pct"]

                try:
                    sim.compute()
                    skor_hasil.append(sim.output["score"])
                except:
                    skor_hasil.append(0.0)

                if i % 50 == 0 or i == total_data - 1:
                    progress_bar.progress((i + 1) / total_data)
                    status_text.text(f"Memproses data {i + 1} dari {total_data}...")

            t_selesai = time.time()
            status_text.text(f"✅ Selesai! Waktu eksekusi: {t_selesai - t_mulai:.2f} detik")

            # Gabungkan Hasil
            res_df = df.copy()
            res_df["Skor_Fuzzy"] = [round(s, 2) for s in skor_hasil]
            res_df["Status_Rekomendasi"] = [tingkat_rekomendasi(s) for s in skor_hasil]

            # Urutkan berdasarkan skor tertinggi
            res_df = res_df.sort_values(by="Skor_Fuzzy", ascending=False).reset_index(drop=True)
            res_df.index = res_df.index + 1

            st.subheader("🏅 Hasil Peringkat Akhir (Top Rekomendasi)")
            
            top_n = st.session_state.top_n
            
            # Kolom yang ditampilkan saja agar tidak penuh
            cols_to_show = ['model', 'price', 'ram_gb', 'storage_gb', 'battery_mah', 'Skor_Fuzzy', 'Status_Rekomendasi']

            def highlight_top(s, n):
                return ["background-color: #22c55e; color: white" if i < n else "" for i in range(len(s))]

            st.dataframe(
                res_df[cols_to_show].style.apply(highlight_top, n=top_n, axis=0),
                use_container_width=True,
            )

            # --- TAMPILKAN GRAFIK ---
            col_pie, col_bar = st.columns(2)
            kategori_counts = (
                res_df["Status_Rekomendasi"]
                .value_counts()
                .reindex(["Buruk", "Cukup", "Baik", "Sangat Baik"], fill_value=0)
            )
            with col_pie:
                fig1, ax1 = plt.subplots()
                warna = ['#ef4444', '#f59e0b', '#3b82f6', '#10b981']
                ax1.pie(kategori_counts, labels=kategori_counts.index, autopct="%1.1f%%", startangle=90, colors=warna)
                ax1.set_title("Proporsi Status Rekomendasi")
                st.pyplot(fig1)
            with col_bar:
                fig2, ax2 = plt.subplots()
                ax2.bar(kategori_counts.index, kategori_counts.values, color=['#ef4444', '#f59e0b', '#3b82f6', '#10b981'])
                ax2.set_title("Distribusi Jumlah Smartphone")
                ax2.set_ylabel("Jumlah Data")
                st.pyplot(fig2)

# ==========================================
# HALAMAN 4: TENTANG PROGRAM
# ==========================================
elif menu == "👥 Tentang Program":
    st.title("👥 Tentang Program SPK Smartphone")

    st.markdown("### Informasi Sistem")
    st.write("- **Tujuan:** Menentukan smartphone terbaik menggunakan metode Fuzzy Inference System (Mamdani)")
    st.write("- **Kriteria Penilaian:** Harga, RAM, Storage, dan Baterai.")
    st.write("- **Pre-processing:** Menggunakan teknik *Percentile Normalization* untuk menyamakan satuan kriteria menjadi skala (0-100).")
    st.write("- **Rule Base:** Otomatis di-generate menggunakan modul `itertools` (kombinasi Rendah, Sedang, Tinggi menghasilkan 81 Aturan logika).")

    st.markdown("---")
    st.markdown("### **Defuzzifikasi**")
    st.write("Sistem ini mendukung 3 jenis perhitungan defuzzifikasi:")
    
    tab1, tab2, tab3 = st.tabs(["📐 Centroid", "✂️ Bisector", "🎯 MOM"])
    with tab1:
        st.markdown("**Centroid (Center of Gravity)**")
        st.markdown("Metode paling umum. Mencari titik berat area, hasil lebih halus dan representatif.")
    with tab2:
        st.markdown("**Bisector**")
        st.markdown("Mencari garis pemisah yang membagi kurva menjadi dua luas yang sama besar.")
    with tab3:
        st.markdown("**MOM (Mean of Maximum)**")
        st.markdown("Hanya mengambil rata-rata puncak kurva yang memiliki probabilitas tertinggi.")