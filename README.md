# Analisis & Forecasting Beban Listrik Panama (2015–2020)

Dashboard interaktif berbasis **Streamlit** dan **streamlit-shadcn-ui** untuk analisis statistik eksploratif, *backtesting* & validasi silang model, dampak kausal COVID-19 (2020), serta proyeksi beban listrik jangka pendek (`7` hari / `168` jam) dan jangka panjang (`12` bulan / `8.760` jam).

## Fitur & Halaman Utama

1. **Ringkasan (`app.py` -> `pages/0_Executive_Overview.py`)** — Ringkasan eksekutif metrik dataset (`48.048` jam), linimasa 3 rezim beban listrik, dan peta 3 stasiun cuaca utama Panama.
2. **Eksplorasi & Audit Data** — Profil harian `24` jam lintas tahun (`2015–2020`), analisis spektral FFT & ACF, uji multikolinearitas VIF, dan *dual-axis explorer*.
3. **Evaluasi Model & CV** — Perbandingan performa 5 model (`Seasonal Naive`, `SARIMA`, `SARIMAX`, `GAM`, `HistGradientBoosting`) dengan uji Diebold-Mariano, *expanding-window* `TimeSeriesSplit` 3-fold, dan audit diagnostik residual.
4. **Cuaca, Libur, & COVID-19** — Kurva sensitivitas non-linear GAM, efek `22` kategori `Holiday_ID`, uji *placebo* `2019`, analisis *counterfactual* `Mar–Jun 2020`, serta *feature importance* (XAI).
5. **Forecast 7 Hari (`168` Jam)** — Proyeksi jangka pendek `HistGradientBoosting` beserta interval prediksi `90%` (`Q05–Q95`) dan peringatan beban puncak interaktif.
6. **Forecast 12 Bulan (`8.760` Jam)** — Proyeksi eksogen murni `GAM Set A` pada skenario Pemulihan Normal vs Tekanan COVID-19 serta simulator pemulihan & anomali suhu.
7. **Simulator What-If** — Simulasi interaktif profil beban `24` jam untuk variasi suhu, kelembapan, kecepatan angin, hari libur, dan kondisi pandemi.

## Struktur Direktori

```text
.
├── app.py                              # Entrypoint Streamlit & st.navigation
├── requirements.txt                    # Dependensi Python (dipin agar kompatibel dengan artefak .pkl)
├── .streamlit/config.toml              # Konfigurasi tema Clean SaaS Light
├── assets/styles.css                   # Tipografi & layout minimal
├── artifacts/                          # Data terproses (.parquet, .json) & model terlatih (.pkl)
├── pages/                              # Halaman-halaman dashboard Streamlit
├── scripts/build_artifacts.py          # Pipeline offline pembentuk artefak dari notebook/data mentah
├── utils/                              # Loader data, visualisasi Matplotlib/Plotly, & inferensi
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
