# Analisis & Forecasting Beban Listrik Panama (2015–2020)

Dashboard interaktif berbasis **Streamlit** dengan gaya desain **Soft UI / Neumorphism** dan struktur *progressive disclosure* (ramah pembaca awam di layar utama, detail statistik tersimpan rapi di laci `expander`) untuk analisis beban listrik nasional Panama, dampak cuaca & COVID-19, perbandingan model, serta simulasi proyeksi ke depan.

## Fitur & Struktur 5 Halaman Utama

1. **Beranda & Ringkasan (`app.py` -> `pages/1_Beranda.py`)** — 4 KPI inti berbahasa sehari-hari, 3 kartu temuan terpenting, grafik tren historis `2015–2020`, serta peta 3 stasiun cuaca utama Panama.
2. **Pola & Faktor Pemicu (`pages/2_Pola_dan_Faktor_Beban.py`)** — Tersusun dalam 3 tab fokus (*Pola Jam & Hari*, *Cuaca & Hari Libur*, *Dampak Pandemi COVID-19*), dengan audit statistik lanjutan (FFT, ACF, VIF, ADF/KPSS, XAI, & 8 titik *blackout*) tersimpan rapi di laci ekspander.
3. **Perbandingan Model (`pages/3_Akurasi_Model.py`)** — Terdiri dari 2 tab: peringkat akurasi 5 model standar di tahun normal (`2019`) serta eksperimen **Transfer Learning Antar-Rezim** (`MLP Fine-Tuning`, `Boosting Warm-Start`, dan `Frozen GAM + Residual Adapter` untuk adaptasi cepat saat pandemi `2020` — sejalan dengan **Tahap 9** di notebook).
4. **Prediksi 7 Hari & 12 Bulan (`pages/4_Prediksi_Kebutuhan.py`)** — 2 tab proyeksi ke depan: operasional **7 hari / 168 jam** (dengan peringatan kapasitas siaga dinamis) dan perencanaan **12 bulan / 8.760 jam** (Skenario Pulih Normal vs Pandemi Berlanjut + pengatur skenario kustom).
5. **Simulasi Coba Sendiri (`pages/5_Simulasi_Interaktif.py`)** — Simulator *What-If* 24 jam dengan 5 tombol skenario cepat 1 klik (*Hari Kerja Normal*, *Cuaca Panas Terik*, *Akhir Pekan Santai*, *Libur Nasional*, *Pembatasan / Pandemi*) serta pengatur suhu, kelembapan, hari libur, sekolah, dan pandemi.

## Struktur Direktori

```text
.
├── app.py                              # Entrypoint Streamlit & st.navigation (5 halaman)
├── requirements.txt                    # Dependensi Python (dipin agar kompatibel dengan artefak .pkl)
├── .streamlit/config.toml              # Konfigurasi tema Soft UI (#EAEFF5 / #3B6EF6)
├── assets/styles.css                   # Styling Neumorphism, tab, expander, kartu KPI, & callout
├── artifacts/                          # Data terproses (.parquet, .json) & model terlatih (.pkl)
├── pages/                              # 5 halaman dashboard berbahasa mudah dipahami
├── scripts/build_artifacts.py          # Pipeline offline pembentuk artefak dari notebook/data mentah
├── utils/                              # Loader data, visualisasi Matplotlib/Plotly Soft UI, & inferensi
├── Analisis_Beban_Listrik_Panama_2015-2020.ipynb
└── Panama_beban_listrik_per_jam_2015-2020_.xlsx
```

## Menjalankan Secara Lokal

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

*(Opsional)* Jika ingin membangun ulang seluruh artefak model dan statistik dari dataset mentah:

```bash
python scripts/build_artifacts.py
```

## Deploy ke Streamlit Community Cloud

1. Pastikan seluruh direktori termasuk `artifacts/` ikut ter-*push* ke repositori GitHub.
2. Di [share.streamlit.io](https://share.streamlit.io/), klik **Create app** lalu pilih repositori dan branch `main`.
3. Isi **Main file path** dengan `app.py`.
4. Pada **Advanced settings**, pilih **Python version `3.12` atau `3.13`** (karena dependensi `numpy==2.5.4` dan `scipy==1.18.1` membutuhkan Python `>=3.12`, dan versi paket dipin agar cocok dengan berkas `.pkl` di `artifacts/`).
