import streamlit as st
import pandas as pd
import numpy as np


# Rumus
def normalize_comparation(m): #normalisasi matriks
    sumCol = m.sum(axis=0)
    norm = m/sumCol
    return norm

def weight(m): #menghitung bobot
    return np.mean(m, axis=1)

def validity_check(m, w):
    n = len(m)
    
    list_RI = {
        2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45,
        10: 1.51, 11: 1.53, 12: 1.54, 13: 1.56, 14: 1.57
    }
    RI = list_RI[n]
    
    CV = m @ w / w
    
    eigen = np.mean(CV)
    
    CI = (eigen - n)/(n-1)
    
    CR = CI/RI
    
    if CR <= 0.1:
        konsisten = 1;
    else:
        konsisten = 0;
    hasil = [eigen, CI, RI, CR, konsisten]
    return hasil
    
        
def final_weight(w_alt, w_crit):
    w_fin = w_alt.T @ w_crit
    return w_fin

def matriks_perbandingan_cost(data):
    MP = np.array([
            [data[0]/data[0],data[1]/data[0],data[2]/data[0],data[3]/data[0],data[4]/data[0]],
            [data[0]/data[1],data[1]/data[1],data[2]/data[1],data[3]/data[1],data[4]/data[1]],
            [data[0]/data[2],data[1]/data[2],data[2]/data[2],data[3]/data[2],data[4]/data[2]],
            [data[0]/data[3],data[1]/data[3],data[2]/data[3],data[3]/data[3],data[4]/data[3]],
            [data[0]/data[4],data[1]/data[4],data[2]/data[4],data[3]/data[4],data[4]/data[4]]
        ])
    return MP

def matriks_perbandingan_benefit(data):
    MP = np.array([
            [data[0]/data[0],data[0]/data[1],data[0]/data[2],data[0]/data[3],data[0]/data[4]],
            [data[1]/data[0],data[1]/data[1],data[1]/data[2],data[1]/data[3],data[1]/data[4]],
            [data[2]/data[0],data[2]/data[1],data[2]/data[2],data[2]/data[3],data[2]/data[4]],
            [data[3]/data[0],data[3]/data[1],data[3]/data[2],data[3]/data[3],data[3]/data[4]],
            [data[4]/data[0],data[4]/data[1],data[4]/data[2],data[4]/data[3],data[4]/data[4]]
        ])
    return MP

# data 
smartphone = ["Samsung Galaxy A5S", "Xiaomi Redmi Note 13 Pro", "iPhone 15", "OPPO Reno 11", "Vivo V29"]
harga = np.array([5.5, 3.8, 13.5, 5.0, 5.2])
baterai = np.array([5000, 5100, 3877, 5000, 4600])
ram = np.array([8, 12, 6, 8, 12])
kamera = np.array([50, 200, 48, 50, 50])

# setup halaman & dataframe alternatif
st.set_page_config(page_title="SPK Smartphone", layout="wide")

df = pd.DataFrame({
    "Smartphone": smartphone,
    "Harga (Juta Rp)": harga,
    "Baterai (mAh)": baterai,
    "RAM (GB)": ram,
    "Kamera (MP)": kamera
})

# Sidebar pengaturan bobot
st.sidebar.title("🌣 Pengaturan")
pilihan = st.sidebar.selectbox(
    "Pilih Halaman : ", 
    ("Page 1 - Data Alternatif", "Page 2 - AHP", "Page 3 - WP")
)

bobot_harga = st.sidebar.slider("Harga (Juta Rp)", min_value=0.1, max_value=1.0, value=0.8, step=0.1)
bobot_baterai = st.sidebar.slider("Baterai (mAh)", min_value=0.1, max_value=1.0, value=0.5, step=0.1)
bobot_ram = st.sidebar.slider("RAM (GB)", min_value=0.1, max_value=1.0, value=0.3, step=0.1)
bobot_kamera = st.sidebar.slider("Kamera (MP)", min_value=0.1, max_value=0.1, value=0.3, step=0.1)

# hitung bobot ternormalisasi
total_bobot = bobot_harga + bobot_baterai + bobot_ram + bobot_kamera
w_harga = bobot_harga / total_bobot
w_baterai = bobot_baterai / total_bobot
w_ram = bobot_ram / total_bobot
w_kamera = bobot_kamera / total_bobot

st.sidebar.subheader("Bobot Ternormalisasi")
st.sidebar.caption(f"Harga: {w_harga:.4f}")
st.sidebar.caption(f"Baterai: {w_baterai:.4f}")
st.sidebar.caption(f"RAM: {w_ram:.4f}")
st.sidebar.caption(f"Kamera: {w_kamera:.4f}")

# navigasi halaman
match pilihan:
    #halaman 1
    case "Page 1 - Data Alternatif":
        st.header("Data Alternatif Smartphone")
        st.write("Sistem pendukung keputusan ini membantu memilih smartphone terbaik berdasarkan empat kriteria utama.")
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.subheader("Keterangan Kriteria")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.error("**Harga (Juta Rp)**\n\nTipe: Cost\n\n*Semakin murah semakin baik*")
        with col2:
            st.success("**Baterai (mAh)**\n\nTipe: Benefit\n\n*Semakin besar semakin baik*")
        with col3:
            st.success("**RAM (GB)**\n\nTipe: Benefit\n\n*Semakin besar semakin baik*")
        with col4:
            st.success("**Kamera (MP)**\n\nTipe: Benefit\n\n*Semakin besar semakin baik*")

    # halaman 2
    case "Page 2 - AHP":
        st.header("Penyelesaian dengan Metode AHP")
        st.subheader("3.1 Matriks Perbandingan Berpasangan")
        
        MPBK = np.array([
            [bobot_harga/bobot_harga, bobot_harga/bobot_baterai, bobot_harga/bobot_ram, bobot_harga/bobot_kamera],
            [bobot_baterai/bobot_harga, bobot_baterai/bobot_baterai, bobot_baterai/bobot_ram, bobot_baterai/bobot_kamera],
            [bobot_ram/bobot_harga, bobot_ram/bobot_baterai, bobot_ram/bobot_ram, bobot_ram/bobot_kamera],
            [bobot_kamera/bobot_harga, bobot_kamera/bobot_baterai, bobot_kamera/bobot_ram, bobot_kamera/bobot_kamera]
        ])

        df_perbandingan_kriteria = pd.DataFrame({
            "Kriteria": ["Harga (Cost)", "Baterai (Benefit)", "RAM (Benefit)", "Kamera (Benefit)"],
            "Harga (Cost)": MPBK[0,:],
            "Baterai (Benefit)": MPBK[1,:],
            "RAM (Benefit)": MPBK[2,:],
            "Kamera (Benefit)": MPBK[3,:]
        })
        st.dataframe(df_perbandingan_kriteria, use_container_width=True, hide_index=True)
        
        st.subheader("3.2 Bobot Prioritas (Eigen Vector)")
        MPBK_norm = normalize_comparation(MPBK)
        wk = weight(MPBK_norm)
        
        df_bobot_prioritas = pd.DataFrame({
            "Kriteria": ["Harga (Cost)", "Baterai (Benefit)", "RAM (Benefit)", "Kamera (Benefit)"],
            "Bobot Prioritas": wk
        })
        st.dataframe(df_bobot_prioritas, use_container_width=True, hide_index=True)
        
        st.subheader("3.3 Nilai Konsistensi (CI, RI, CR)")
        hasil_konsistensi = validity_check(MPBK, wk)
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.write("λ maks")
            st.subheader(f"{hasil_konsistensi[0]:.4f}")
        with col2:
            st.write("CI")
            st.subheader(f"{hasil_konsistensi[1]:.4f}")
        with col3:
            st.write("RI")
            st.subheader(f"{hasil_konsistensi[2]:.4f}")
        with col4:
            st.write("CR")
            st.subheader(f"{hasil_konsistensi[3]:.4f}")
        
        st.subheader("3.4 Keterangan Konsistensi")
        if hasil_konsistensi[4] == 1:
            st.success("Konsisten ദ്ദി(˵ •̀ ᴗ - ˵ ) ✧")
        else:
            st.error("Tidak Konsisten .·°՞(っ-ᯅ-ς)՞°·.")
        
        st.subheader("3.5 Skor Akhir dan Ranking")
        
        # hitung AHP tiap alternatif secara efisien
        wa = weight(normalize_comparation(matriks_perbandingan_cost(harga)))
        wb = weight(normalize_comparation(matriks_perbandingan_benefit(baterai)))
        wc = weight(normalize_comparation(matriks_perbandingan_benefit(ram)))
        wd = weight(normalize_comparation(matriks_perbandingan_benefit(kamera)))
        
        w_total = np.array([wa, wb, wc, wd])
        w_final = final_weight(w_total, wk)
        
        # proses perangkingan
        df_perangkingan = pd.DataFrame({
            "Smartphone": smartphone,
            "Skor AHP": w_final
        }).sort_values(by="Skor AHP", ascending=False).reset_index(drop=True)
        
        df_perangkingan.index += 1
        df_perangkingan = df_perangkingan.reset_index().rename(columns={"index": "Ranking"})
        
        st.dataframe(df_perangkingan, use_container_width=True, hide_index=True)
        
        pemenang_nama = df_perangkingan.iloc[0]["Smartphone"]
        pemenang_skor = df_perangkingan.iloc[0]["Skor AHP"]
        st.success(f" **Kesimpulan AHP:** ✧｡◝(ᵔᗜᵔ)◜✧*｡\n\nBerdasarkan metode AHP, smartphone terbaik adalah **{pemenang_nama}** dengan skor **{pemenang_skor:.4f}**")

    # halaman 3
    case "Page 3 - WP":
        st.header("Penyelesaian dengan Metode WP")
        
        st.subheader("4.1 Bobot Kriteria (W)")
        df_bobot_wp = pd.DataFrame({
            "Kriteria": ["Harga (Cost)", "Baterai (Benefit)", "RAM (Benefit)", "Kamera (Benefit)"],
            "Bobot Awal": [bobot_harga, bobot_baterai, bobot_ram, bobot_kamera],
            "Bobot Ternormalisasi": [w_harga, w_baterai, w_ram, w_kamera]
        })
        st.dataframe(df_bobot_wp, use_container_width=True, hide_index=True)

        st.subheader("4.2 Vektor S")
        st.latex(r"S_i = \prod_{j=1}^{n} X_{ij}^{w_j}")

        # pangkat cost dibikin negatif, benefit dibikin positif
        pangkat_w = np.array([-w_harga, w_baterai, w_ram, w_kamera])
        
        vektor_S = []
        for i in range(len(smartphone)):
            # rumus vektor S disederhanakan dalam satu baris
            s_val = (harga[i]**pangkat_w[0]) * (baterai[i]**pangkat_w[1]) * (ram[i]**pangkat_w[2]) * (kamera[i]**pangkat_w[3])
            vektor_S.append(s_val)
            
        vektor_S = np.array(vektor_S)
        
        df_vektor_s = pd.DataFrame({
            "Smartphone": smartphone,
            "Vektor S": vektor_S
        })
        st.dataframe(df_vektor_s, use_container_width=True, hide_index=True)
        
        st.subheader("4.3 Nilai Akhir (Vektor V) dan Ranking")
        st.latex(r"V_i = \frac{S_i}{\sum S}")
        
        vektor_V = vektor_S / np.sum(vektor_S)
        
        df_ranking_wp = pd.DataFrame({
            "Smartphone": smartphone,
            "Vektor V": vektor_V,
        }).sort_values(by="Vektor V", ascending=False).reset_index(drop=True)
        
        df_ranking_wp.index += 1
        df_ranking_wp = df_ranking_wp.reset_index().rename(columns={"index": "Ranking"})
        
        st.dataframe(df_ranking_wp, use_container_width=True, hide_index=True)
        
        pemenang_wp = df_ranking_wp.iloc[0]["Smartphone"]
        skor_wp = df_ranking_wp.iloc[0]["Vektor V"]
        st.success(f" **Kesimpulan WP:** ✧｡◝(ᵔᗜᵔ)◜✧*｡\n\nBerdasarkan metode WP, rekomendasi utama adalah **{pemenang_wp}** dengan nilai **{skor_wp:.4f}**")