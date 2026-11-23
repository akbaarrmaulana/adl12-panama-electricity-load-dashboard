# Rancangan Analisis Data & Tinjauan Literatur: Beban Listrik Per Jam Panama (2015–2020)

Dokumen ini disusun sebagai panduan lengkap untuk memahami karakteristik dataset [Panama_beban_listrik_per_jam_2015-2020_.xlsx](file:///c:/Users/user/Documents/S2/Semester%202/Analisis%20Data%20Lanjut/W12%20adl%201/Panama_beban_listrik_per_jam_2015-2020_.xlsx), menelusuri literatur ilmiah yang menggunakan dataset ini maupun dataset dengan pola serupa, serta memetakan berbagai rancangan analisis yang terbuka untuk penelitian atau tugas analisis data tingkat lanjut.

---

## 1. Ringkasan Eksekutif

Dataset `Panama_beban_listrik_per_jam_2015-2020_.xlsx` merupakan versi terjemahan berbahasa Indonesia dari dataset benchmark internasional ***"Short-Term Electricity Load Forecasting (Panama Case Study)"*** yang dikompilasi oleh Ernesto Aguilar Madrid (tersedia di Kaggle dan Mendeley Data). Data ini menggabungkan:

1. **Beban listrik nasional historis (*post-dispatch*)** dari operator jaringan nasional Panama (*Centro Nacional de Despacho*, CND).
2. **Data meteorologi per jam** dari NASA Earthdata untuk tiga wilayah/kota utama di Panama: **Tocumen (Kota Panama)**, **Santiago**, dan **David**.
3. **Atribut kalender kontekstual** berupa indikator hari libur nasional (`holiday`, `Holiday_ID`) serta periode sekolah (`school`).

Secara struktur, dataset ini sangat bersih (tanpa *missing values* dan berinterval tepat 1 jam), tetapi menyimpan fenomena dunia nyata yang sangat kaya untuk dimodelkan—mulai dari musiman ganda (*multi-seasonality* harian, mingguan, tahunan), sensitivitas non-linear terhadap suhu tropis, efek hari libur spesifik, anomali pemadaman listrik massal (*blackout*), hingga *structural break* akibat kebijakan pembatasan pandemi COVID-19 pada Maret–Juni 2020.

---

## 2. Profil & Temuan Awal Dataset (Berdasarkan Inspeksi Langsung)

Workbook terdiri atas dua sheet:
- **`data`**: `48.048` baris × `17` kolom (setara tepat `2.002` blok hari × 24 jam).
- **`deskripsi`**: `17` baris × `3` kolom (`kolom`, `deskripsi`, `satuan`) berisi metadata setiap variabel.

Rentang waktu observasi berlangsung dari **`2015-01-03 01:00:00`** hingga **`2020-06-27 00:00:00`** (zona waktu Panama UTC−05:00) dengan resolusi tepat 1 jam (`48.047` selisih waktu bernilai `01:00:00`, tanpa celah/duplikasi) dan **0 missing values** pada seluruh 17 kolom.

### 2.1 Ringkasan Variabel

| Kelompok | Nama Kolom | Tipe | Deskripsi & Satuan (Sheet `deskripsi`) | Statistik Deskriptif (48.048 observasi) |
| :--- | :--- | :--- | :--- | :--- |
| **Waktu** | `datetime` | `datetime64` | Tanggal-waktu, UTC−05:00 (`tanggal-waktu`) | `2015-01-03 01:00` s.d. `2020-06-27 00:00` |
| **Target** | `nat_demand` | `float64` | Beban listrik nasional (`MWh`) | Mean `1.182,87`; Std `192,07`; Min `85,19`; Median `1.168,43`; Max `1.754,88` |
| **Cuaca: Tocumen** | `T2M_toc` | `float64` | Suhu 2 m di Tocumen, Kota Panama (`°C`) | Mean `27,40`; Std `1,68`; Min `22,95`; Median `27,12`; Max `35,04` |
| | `QV2M_toc` | `float64` | Kelembapan relatif 2 m (`%` menurut README)* | Mean `0,0183`; Std `0,0016`; Min `0,0121`; Median `0,0186`; Max `0,0227` |
| | `TQL_toc` | `float64` | Presipitasi cair (`l/m²` menurut README) | Mean `0,0800`; Std `0,0656`; Min `0,0000`; Median `0,0652`; Max `0,5212` |
| | `W2M_toc` | `float64` | Kecepatan angin 2 m (`m/s` menurut README) | Mean `13,39`; Std `7,30`; Min `0,009`; Median `12,18`; Max `39,23` |
| **Cuaca: Santiago** | `T2M_san` | `float64` | Suhu 2 m di Kota Santiago (`°C`) | Mean `26,92`; Std `3,02`; Min `19,77`; Median `26,17`; Max `39,06` |
| | `QV2M_san` | `float64` | Kelembapan relatif 2 m (`%` menurut README)* | Mean `0,0178`; Std `0,0019`; Min `0,0102`; Median `0,0184`; Max `0,0222` |
| | `TQL_san` | `float64` | Presipitasi cair (`l/m²` menurut README) | Mean `0,1063`; Std `0,0863`; Min `0,000009`; Median `0,0860`; Max `0,4850` |
| | `W2M_san` | `float64` | Kecepatan angin 2 m (`m/s` menurut README) | Mean `7,05`; Std `4,10`; Min `0,060`; Median `5,99`; Max `24,48` |
| **Cuaca: David** | `T2M_dav` | `float64` | Suhu 2 m di Kota David (`°C`) | Mean `24,72`; Std `2,41`; Min `19,93`; Median `24,00`; Max `34,22` |
| | `QV2M_dav` | `float64` | Kelembapan relatif 2 m (`%` menurut README)* | Mean `0,0169`; Std `0,0016`; Min `0,0097`; Median `0,0172`; Max `0,0211` |
| | `TQL_dav` | `float64` | Presipitasi cair (`l/m²` menurut README) | Mean `0,1447`; Std `0,0879`; Min `0,000032`; Median `0,1302`; Max `0,4778` |
| | `W2M_dav` | `float64` | Kecepatan angin 2 m (`m/s`; README menulis `'m/2'`) | Mean `3,57`; Std `1,71`; Min `0,015`; Median `3,41`; Max `10,29` |
| **Kalender** | `Holiday_ID` | `int64` | ID jenis hari libur (`0` = bukan libur) | 23 nilai unik (`0` s.d. `22`); Mean `0,705` |
| | `holiday` | `int64` | Indikator hari libur (`1` = libur, `0` = biasa) | 6,29% libur (`3.024` jam); 93,71% biasa (`45.024` jam) |
| | `school` | `int64` | Indikator masa sekolah (`1` = sekolah, `0` = libur) | 72,78% masa sekolah (`34.969` jam); 27,22% libur (`13.079` jam) |

> *Catatan kualitas metadata tertera pada poin 2.2.*

### 2.2 Temuan Empiris Kunci Sebagai Dasar Rancangan Analisis

1. **Dominasi Korelasi Suhu (`T2M`)**
   Korelasi Pearson linear terhadap `nat_demand` menunjukkan suhu udara merupakan prediktor cuaca paling dominan di ketiga kota:
   - `T2M_toc`: **r = 0,6528**
   - `T2M_dav`: **r = 0,6483**
   - `T2M_san`: **r = 0,6270**
   - Variabel cuaca lain berkorelasi jauh lebih lemah secara linear: `W2M_san` (r = 0,1918), `TQL_san` (r = 0,1190), `W2M_toc` (r = 0,0984), `TQL_toc` (r = 0,0731), `TQL_dav` (r = 0,0420), `QV2M_san` (r = 0,0222), `QV2M_dav` (r = −0,0021), `QV2M_toc` (r = −0,0367), `W2M_dav` (r = −0,0548).
   - Temuan ini mengindikasikan dominannya beban pendingin ruangan (AC) di iklim tropis Panama serta adanya potensi **multikolinearitas tinggi** antar fitur suhu ketiga kota.

2. **Efek Kalender (`holiday` dan `school`)**
   - **`holiday`**: Rata-rata beban saat hari biasa (`holiday = 0`) adalah **1.191,12 MWh** (std `192,68`), sedangkan pada hari libur (`holiday = 1`) turun menjadi **1.060,09 MWh** (std `131,60`)—penurunan sekitar **−131,03 MWh (−11,0%)** dengan variabilitas yang lebih sempit (korelasi r = −0,1657; `Holiday_ID` r = −0,1298).
   - **`school`**: Rata-rata beban selama masa sekolah (`school = 1`) sedikit lebih tinggi, yakni **1.187,57 MWh** (std `192,77`), dibanding masa libur sekolah (`school = 0`) sebesar **1.170,29 MWh** (std `189,60`; korelasi r = 0,0400).

3. **Tren Pertumbuhan Konsisten 2015–2019**
   Rata-rata beban tahunan tumbuh stabil dari tahun ke tahun sebelum pandemi:
   - **2015** (`n = 8.711`): **1.131,53 MWh** (std `194,15`)
   - **2016** (`n = 8.784`): **1.157,19 MWh** (std `190,29`)
   - **2017** (`n = 8.760`): **1.182,32 MWh** (std `188,96`)
   - **2018** (`n = 8.760`): **1.205,68 MWh** (std `185,45`)
   - **2019** (`n = 8.760`): **1.243,58 MWh** (std `189,14`)
   - **2020** (`n = 4.273`, Jan–Jun): turun kembali ke **1.170,23 MWh** (std `176,12`), meskipun puncak maksimum absolut seluruh dataset tetap tercatat di 2020 (`1.754,88 MWh`).

4. **Structural Break COVID-19 (Maret–Juni 2020)**
   Perbandingan *year-over-year* (2019 vs 2020) menunjukkan kejutan permintaan yang sangat jelas sejak lockdown Maret–April 2020:

   | Bulan | Rata-rata 2019 (MWh) | Rata-rata 2020 (MWh) | Perubahan YoY |
   | :--- | :---: | :---: | :---: |
   | Maret | 1.238,63 | 1.208,99 | −2,4% |
   | April | 1.277,86 | 1.062,04 | **−16,9%** |
   | Mei | 1.277,86 | 1.088,15 | **−14,8%** |
   | Juni | 1.266,55 | 1.109,26 | **−12,4%** |

5. **Anomali Ekstrem (Indikasi Gangguan / *Blackout*)**
   Terdapat **8 observasi** dengan `nat_demand < 600 MWh` (semuanya terjadi pada hari non-libur, `holiday = 0`, `Holiday_ID = 0`):
   - **20 Januari 2019 pukul 12:00–16:00** (5 jam berturut-turut): anjlok drastis hingga **85,19 MWh** (`12:00`), lalu bertahap pulih ke `135,76` (`13:00`), `263,62` (`14:00`), `417,45` (`15:00`), dan `475,80 MWh` (`16:00`)—sesuai catatan peristiwa *blackout* nasional di Panama pada Januari 2019.
   - Penurunan tajam terisolasi lainnya: **`2015-01-24 17:00`** (`509,84 MWh`), **`2017-06-27 12:00`** (`380,59 MWh`), dan **`2017-07-01 14:00`** (`560,70 MWh`).

6. **Catatan Kualitas Metadata (`QV2M_*` dan `W2M_dav`)**
   - Nilai `QV2M_toc`, `QV2M_san`, dan `QV2M_dav` berada di rentang **0,0097 – 0,0227**. Pada konvensi data meteorologi NASA MERRA-2, `QV2M` sebenarnya adalah ***specific humidity* (kelembapan spesifik dalam kg/kg)**, meskipun sheet `deskripsi` (mengutip README asli dataset) menuliskannya sebagai kelembapan relatif dalam satuan `%`.
   - Satuan `W2M_dav` pada sheet `deskripsi` memuat salah ketik bawaan dari README asli (`m/2`, seharusnya `m/s`).
   - File ini tidak menyertakan kolom prakiraan resmi operator CND (*pre-dispatch forecast*) yang ada di sebagian file terpisah pada repositori Mendeley/Kaggle asli, sehingga pemodelan berfokus murni pada data aktual beban, cuaca, dan kalender.

---

## 3. Referensi Paper Terkait

Literatur terbagi menjadi dua kelompok: **(A)** penelitian yang menggunakan dataset Panama ini secara langsung, dan **(B)** penelitian dengan pola data serta pendekatan analisis serupa (beban listrik per jam + cuaca + kalender + guncangan COVID-19).

### A. Paper yang Menggunakan Dataset Beban Listrik Panama Secara Langsung

1. **Aguilar Madrid, E., & António, N. (2021).** *Short-Term Electricity Load Forecasting with Machine Learning.* **Information** (MDPI), 12(2), 50. DOI: [10.3390/info12020050](https://doi.org/10.3390/info12020050)
   - **Kontribusi:** Paper rujukan utama yang memperkenalkan dataset ini. Menguji peramalan horizon 1 minggu ke depan (`168` jam) menggunakan *Multiple Linear Regression* (MLR), *k-Nearest Neighbors* (KNN), $\varepsilon$-*Support Vector Regression* ($\varepsilon$-SVR), *Random Forest* (RF), dan *XGBoost*.
   - **Relevansi:** Membandingkan performa model ML terhadap prakiraan resmi operator (CND) baik pada kondisi normal maupun saat guncangan awal lockdown COVID-19 Maret–Juni 2020.

2. ***Short-term power load forecasting based on the CEEMDAN-TCN-ESN model*** **(2023, PubMed Central / PMC).**
   - **Kontribusi:** Menggunakan data `nat_demand` Panama (subset 2018–2019, 17.520 titik per jam) untuk membangun arsitektur hibrida dekomposisi sinyal *Complete Ensemble Empirical Mode Decomposition with Adaptive Noise* (CEEMDAN) yang dipadukan dengan *Temporal Convolutional Network* (TCN) dan *Echo State Network* (ESN).
   - **Relevansi:** Menunjukkan efektivitas pemecahan deret waktu beban ke beberapa komponen frekuensi (IMF) sebelum dimodelkan oleh *deep learning*.

3. ***Short-Term Load Forecasting Using ANN Model*** **(2024, ResearchGate).**
   - **Kontribusi:** Menerapkan *Artificial Neural Network* (ANN) / *Multilayer Perceptron* (MLP) untuk memprediksi `nat_demand` per jam menggunakan fitur cuaca kontinu dan kalender dari dataset terbuka Panama.

4. ***An Informer Model for Very Short-Term Power Load Forecasting*** **(2025, ResearchGate).**
   - **Kontribusi:** Menguji arsitektur *Transformer* efisien (*Informer*) pada keseluruhan `48.048` observasi dataset Panama 2015–2020 untuk peramalan beban jangka sangat pendek (*very short-term load forecasting*, VSTLF).

5. ***Forecasting very short-term power load with hybrid interpretable deep models*** **(2025, Taylor & Francis).**
   - **Kontribusi:** Mengembangkan model *deep learning* hibrida yang tetap mempertahankan interpretabilitas (*interpretable deep models*) pada dataset Panama (`nat_demand` + fitur meteorologi multikota).

6. ***Machine Learning for Forecasting Electricity Demand in Emerging Economies Amid COVID-19 Uncertainty*** **(MDPI).**
   - **Kontribusi:** Berfokus pada dinamika permintaan listrik di Panama selama ketidakpastian pandemi COVID-19 dengan memadukan *Convolutional Neural Networks* (CNN) dan fitur multi-skala, serta menyertakan analisis **SHAP** untuk menjelaskan dampak perubahan mobilitas terhadap pergeseran beban.

---

### B. Paper dengan Pola Data & Analisis Serupa (Hourly Load + Cuaca + Kalender + COVID-19)

1. **Hong, T., & Fan, S. (2016).** *Probabilistic electric load forecasting: A tutorial review.* **International Journal of Forecasting**, 32(3), 914–938 (terkait kompetisi **GEFCom2014**).
   - **Pola Analisis:** Panduan fundamental peramalan probabilistik (kuantil/densitas) pada data beban listrik per jam + suhu + kalender menggunakan *quantile regression*, model aditif, dan metrik *pinball loss*.

2. **Obst, D., de Vilmarest, J., & Goude, Y. (2021).** *Adaptive Methods for Short-Term Electricity Load Forecasting During COVID-19 Lockdown in France.* **IEEE Transactions on Power Systems**.
   - **Pola Analisis:** Mengatasi *concept drift* mendadak saat lockdown COVID-19 dengan **Generalized Additive Models (GAM)** yang diadaptasi secara *online* menggunakan *Kalman Filter* untuk memperbarui bobot efek cuaca dan kalender secara dinamis.

3. **Ziel, F.** *Smoothed Bernstein Online Aggregation for Day-Ahead Electricity Demand Forecasting* (terkait *IEEE DataPort Competition: Post-COVID Paradigm*).
   - **Pola Analisis:** Strategi kombinasi model (*online aggregation*) pada data beban per jam pasca-COVID dengan menggabungkan GAM, regresi *Lasso* berdimensi tinggi, penyesuaian hari libur, dan pemetaan efek cuaca non-linear.

4. **Wakashiro.** *Causal Impact of Severe Events on Electricity Demand: The case of COVID-19 in Japan.*
   - **Pola Analisis:** Estimasi dampak kausal kebijakan darurat COVID-19 menggunakan **Bayesian Structural Time Series (CausalImpact)** untuk membangun deret kontrafaktual bebas-pandemi yang dikontrol oleh suhu, musiman, dan hari libur.

5. **Berezvai, Z., Hortay, O., & Szőke, T.** *The impact of COVID-19 measures on intraday electricity load curves in the European Union: A panel approach.* **Sustainable Energy, Grids and Networks**.
   - **Pola Analisis:** Analisis ekonometrika panel terhadap **kurva beban intraharian** (24 jam) untuk melihat bagaimana tingkat keketatan pembatasan sosial mengubah profil beban pagi, siang, dan malam antara hari kerja dan akhir pekan.

6. ***The impact of the COVID-19 on households' hourly electricity consumption in Canada*** **(Applied Energy / PMC).**
   - **Pola Analisis:** Mengombinasikan ***k-means clustering*** pada profil beban per jam dan ***changepoint analysis*** untuk memisahkan komponen beban yang bergantung pada suhu dari perubahan perilaku konsumen selama lockdown.

---

## 4. Rancangan Analisis yang Terbuka (7 Jalur Riset)

Berdasarkan struktur variabel dan temuan empiris di Bagian 2, berikut tujuh jalur rancangan analisis yang dapat dipilih (secara tunggal maupun dikombinasikan):

---

### Jalur 1: Peramalan Beban Listrik Jangka Pendek & Sangat Pendek (STLF / VSTLF)

* **Pertanyaan Riset:** Seberapa akurat model statistik klasik, *machine learning* tabular, *deep learning* sekuensial, dan model hibrida dekomposisi dalam memprediksi `nat_demand` pada horizon 1 jam (VSTLF), 24 jam (*day-ahead*), dan 168 jam (1 minggu STLF)?
* **Metode / Model (Perbandingan Bertingkat):**
  1. *Baseline Naif Musiman:* $\hat{y}_t = y_{t-24}$ atau $\hat{y}_t = y_{t-168}$ (minggu lalu pada jam yang sama).
  2. *Statistik / Time Series Klasik:* **SARIMAX**, **TBATS** (unggul untuk musiman ganda harian + mingguan + tahunan), atau **Prophet** dengan regresor cuaca.
  3. *Machine Learning Tabular:* **XGBoost**, **LightGBM**, **CatBoost**, **Random Forest**, $\varepsilon$-**SVR**.
  4. *Deep Learning Sekuensial:* **LSTM / GRU**, **TCN**, **Informer** / **Temporal Fusion Transformer (TFT)**.
  5. *Hibrida Dekomposisi Sinyal:* **STL** atau **CEEMDAN** untuk memecah sinyal `nat_demand` menjadi tren, musiman, dan residu sebelum diprediksi masing-masing oleh ML/DL.
* **Feature Engineering & Variabel:**
  - *Lag target:* `t-1`, `t-2`, `t-24`, `t-48`, `t-168` (disesuaikan dengan horizon prediksi; lihat Bagian 5).
  - *Statistik bergerak (rolling):* rata-rata, min, maks, dan simpangan baku 24 jam dan 168 jam terakhir.
  - *Fitur kalender & siklikal:* `sin`/`cos` untuk jam (`24`), hari dalam minggu (`7`), hari dalam tahun (`365.25`), indikator akhir pekan, `holiday`, `Holiday_ID`, dan `school`.
  - *Fitur cuaca:* suhu/kelembapan/angin/presipitasi ketiga kota + *cooling degree* (mis. $\max(0, T2M - 25)$) dan interaksi `T2M × jam`.
* **Metrik Evaluasi:** **MAE** (MWh), **RMSE** (MWh), **MAPE** (%), dan **sMAPE** (%).
* **Catatan Penerapan:** Evaluasi wajib dipisah antara rezim normal (mis. uji tahun 2019) dan rezim disrupsi (Maret–Juni 2020) agar performa model tidak tersamarkan oleh *structural break* COVID-19.

---

### Jalur 2: Peramalan Probabilistik & Kuantifikasi Ketidakpastian (*Probabilistic Load Forecasting*)

* **Pertanyaan Riset:** Bagaimana membangun interval prediksi (*prediction interval*) beban listrik per jam yang terkalibrasi baik untuk mendukung manajemen risiko cadangan daya (*spinning reserve*), khususnya ketika variabilitas meningkat saat cuaca ekstrem atau transisi pandemi?
* **Metode / Model:**
  - **Quantile Regression** (linear / GAM kuantil) dan **Quantile Gradient Boosting** (LightGBM/XGBoost dengan fungsi objektif *pinball/quantile loss* untuk kuantil $\tau \in \{0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95\}$).
  - **Conformal Prediction** (*Conformalized Quantile Regression* / *Adaptive Conformal Inference*) untuk menjamin cakupan interval secara bebas-distribusi (*distribution-free*) bahkan saat terjadi pergeseran distribusi di tahun 2020.
  - **DeepAR / TFT** yang langsung mengestimasi parameter distribusi atau multi-kuantil.
* **Feature Engineering & Variabel:** Sama seperti Jalur 1, ditambah fitur volatilitas beban historis (`rolling std` 24 jam dan 168 jam) serta indikator ketidakpastian cuaca antar-kota (selisih/rentang `T2M` antar Tocumen–Santiago–David).
* **Metrik Evaluasi:**
  - **Pinball Loss** (rata-rata lintas seluruh kuantil).
  - **PICP (*Prediction Interval Coverage Probability*)** — memeriksa apakah cakupan empiris mendekati target nominal (mis. 90%).
  - **Winkler Score** atau **CRPS (*Continuous Ranked Probability Score*)** — menyeimbangkan cakupan dan ketajaman lebar interval.
* **Catatan Penerapan:** Pendekatan *Adaptive Conformal Inference* sangat relevan diuji pada data Maret–Juni 2020 untuk melihat seberapa cepat lebar interval beradaptasi ketika error model melonjak akibat lockdown.

---

### Jalur 3: Analisis Dampak Kausal & *Structural Break* COVID-19 (Maret–Juni 2020)

* **Pertanyaan Riset:** Berapa besar penurunan beban listrik murni akibat kebijakan pembatasan COVID-19 pada Maret–Juni 2020 setelah efek cuaca, tren pertumbuhan tahunan, dan kalender dikendalikan? Pada jam berapa perubahan paling ekstrem terjadi?
* **Metode / Model:**
  - **Counterfactual Forecasting (Pendekatan Prediktif Kausal):** Latih model berakurasi tinggi (GAM, LightGBM, atau XGBoost) hanya pada data pra-pandemi (**Januari 2015 – Februari 2020**). Gunakan nilai aktual cuaca dan kalender **Maret–Juni 2020** untuk memprediksi beban kontrafaktual $\hat{Y}_{t}^{(0)}$ ("beban seandainya tidak ada COVID-19"), lalu hitung dampak kausal $\Delta_t = Y_t^{\text{aktual}} - \hat{Y}_t^{(0)}$.
  - **Bayesian Structural Time Series (BSTS / `CausalImpact`):** Mengestimasi deret kontrafaktual beserta interval kredibel 95% dengan variabel kontrol `T2M_*`, `QV2M_*`, `holiday`, dan `school`.
  - **Interrupted Time Series (ITS) & Deteksi Changepoint:** Uji regresi terputus (*segmented regression*) atau algoritma deteksi *changepoint* (**PELT**, **CUSUM**, **Bai-Perron**) untuk mengidentifikasi tanggal persis mulainya patahan struktural dan fase pemulihan bertahap dari April (rata-rata −16,9% YoY) menuju Juni (−12,4% YoY).
* **Feature Engineering & Variabel:** Khusus pada skema kontrafaktual, **hindari menggunakan lag beban jangka pendek (`t-1`, `t-24`)** saat melakukan prediksi sepanjang Maret–Juni 2020, karena lag tersebut sudah terkontaminasi oleh penurunan akibat pandemi! Gunakan hanya fitur eksogen: tren waktu (`time_index`), fitur siklikal waktu, `holiday`, `Holiday_ID`, `school`, dan cuaca 3 kota.
* **Metrik Evaluasi:**
  - Validitas model kontrafaktual pada periode placebo pra-pandemi (mis. Maret–Juni 2019): RMSE, MAPE, bias rata-rata (harus mendekati 0).
  - Estimasi dampak kausal: penurunan absolut rata-rata (MWh/jam), persentase penurunan relatif (%), total energi yang "hilang" (GWh kumulatif), dan *credible interval* 95%.
* **Catatan Penerapan:** Agregasi $\Delta_t$ berdasarkan jam (00:00–23:00) dan tipe hari (hari kerja vs akhir pekan) memungkinkan analisis pergeseran jam puncak (*peak load shifting*) akibat *work-from-home*.

---

### Jalur 4: Pemodelan Sensitivitas Cuaca Non-Linear & Efek Kalender (GAM & Ekonometrika)

* **Pertanyaan Riset:** Bagaimana bentuk hubungan non-linear suhu, kelembapan, curah hujan, dan kecepatan angin terhadap konsumsi listrik di tiga kota Panama? Seberapa besar perbedaan penurunan beban di antara 22 kategori hari libur (`Holiday_ID` 1–22)?
* **Metode / Model:**
  - **Generalized Additive Models (GAM):**
    $$y_t = \beta_0 + f_{\text{tren}}(t) + f_{\text{jam}}(\text{hour}_t) + f_{\text{suhu}}(T2M_t) + f_{\text{interaksi}}(T2M_t, \text{hour}_t) + \sum_{k=1}^{22}\gamma_k \mathbb{I}(\text{Holiday\_ID}_t = k) + \delta \cdot \text{school}_t + \varepsilon_t$$
  - **Regresi Kuantil / Ekonometrika Interaksi:** Memodelkan efek marjinal suhu pada persentil beban rendah vs beban puncak.
  - **Penanganan Multikolinearitas Cuaca 3 Kota:** Principal Component Analysis (PCA) untuk mengekstrak faktor suhu/kelembapan nasional, atau indeks tertimbang berbasis kota.
* **Feature Engineering & Variabel:**
  - Encoding kategorikal penuh untuk `Holiday_ID` (0–22) sebagai faktor diskret, bukan numerik kontinu.
  - Interaksi `hour × is_weekend` dan `hour × T2M` karena sensitivitas suhu di siang hari kerja berbeda dengan dini hari atau hari libur.
  - Transformasi non-linear untuk presipitasi (`TQL_*`, yang sangat miring ke kanan / *right-skewed*).
* **Metrik Evaluasi:** Adjusted $R^2$, *Deviance Explained* (%), **AIC / BIC**, GCV (*Generalized Cross-Validation*), serta signifikansi statistik dan *Confidence Interval* dari kurva *partial effect* spline.
* **Catatan Penerapan:** GAM sangat unggul untuk keperluan inferensi ilmiah karena transparan, mampu memvisualisasikan kurva respons suhu secara langsung, dan dapat dijadikan baseline interpretabel sebelum membandingkan dengan ML *black-box*.

---

### Jalur 5: Deteksi Anomali & Identifikasi Gangguan Sistem (*Blackout / Outlier Detection*)

* **Pertanyaan Riset:** Dapatkah algoritma deteksi anomali tanpa supervisi (*unsupervised*) maupun berbasis residu peramalan mendeteksi secara otomatis gangguan sistem tenaga listrik (seperti *blackout* 20 Januari 2019) serta membedakannya dari hari libur nasional yang memang berbeban rendah?
* **Metode / Model:**
  - **Berbasis Dekomposisi & Statistik:** Dekomposisi **STL / MSTL** (*Multiple Seasonal-Trend decomposition using Loess*) lalu mendeteksi nilai ekstrem pada komponen residu menggunakan **Modified Z-Score (MAD)**, **IQR**, atau batas kontrol adaptif (EWMA).
  - **Berbasis Pembelajaran Mesin Unsupervised:** **Isolation Forest**, **Local Outlier Factor (LOF)**, atau **One-Class SVM** pada ruang fitur multi-dimensi (beban + residu + suhu + jam + `holiday`).
  - **Berbasis Rekonstruksi Deep Learning:** **LSTM Autoencoder** atau **Dense Autoencoder** yang dilatih pada periode operasi normal untuk mendeteksi anomali berdasarkan *reconstruction error* yang melampaui ambang persentil (mis. P99,9).
* **Feature Engineering & Variabel:** Selain `nat_demand` mentah, fitur krusial adalah **selisih perubahan per jam** ($\Delta y_t = y_t - y_{t-1}$), **deviasi terhadap median jam & tipe hari yang sama** ($y_t - \tilde{y}_{\text{hour, dow}}$), serta konteks suhu (`T2M_toc`) dan `holiday`.
* **Metrik Evaluasi:**
  - Studi kasus verifikasi terarah (*case-study recall*) pada 8 titik anjlok `< 600 MWh` yang teridentifikasi (terutama peristiwa **20 Januari 2019 jam 12:00–16:00** dengan minimum `85,19 MWh`, serta **2015-01-24 17:00**, **2017-06-27 12:00**, dan **2017-07-01 14:00**).
  - *False Positive Rate* pada hari libur resmi (`holiday = 1`) untuk memastikan model tidak salah mengira hari libur sebagai gangguan jaringan.
* **Catatan Penerapan:** Jika anomali dianalisis pada keseluruhan rentang 2015–2020, bedakan antara **point/collective blackout anomaly** (turun mendadak beberapa jam lalu kembali normal) dan **regime shift** (penurunan persisten berbulan-bulan pada April–Juni 2020).

---

### Jalur 6: Klasterisasi & *Profiling* Kurva Beban Harian (*Unsupervised Learning*)

* **Pertanyaan Riset:** Berapa banyak arketipe bentuk kurva beban listrik 24 jam (*intraday load profiles*) yang ada di Panama selama 2.002 hari, dan bagaimana komposisi klaster tersebut berubah antar hari kerja, akhir pekan, hari libur, musim hujan/kemarau, serta periode lockdown 2020?
* **Metode / Model:**
  1. **Reshape & Normalisasi:** Ubah deret waktu `48.048` jam menjadi matriks `2.002 hari × 24 jam`. Lakukan dua eksperimen terpisah: (a) **skala MWh absolut** (menangkap level dan bentuk), dan (b) **profil ternormalisasi** (z-score harian atau dibagi total/rata-rata harian untuk menangkap bentuk kurva murni tanpa bias tren tahunan).
  2. **Algoritma Klasterisasi:** **K-Means**, **Gaussian Mixture Models (GMM)**, **Agglomerative Hierarchical Clustering**, dan **Time-Series K-Means dengan Dynamic Time Warping (DTW)** untuk mengakomodasi pergeseran kecil jam puncak.
  3. **Analisis Fungsional:** **Functional Principal Component Analysis (FPCA)** untuk mereduksi kurva 24 jam menjadi skor komponen utama harian guna memvisualisasikan lintasan transisi sebelum vs selama lockdown.
* **Feature Engineering & Variabel:** Vektor 24-dimensi beban per jam (`h00`–`h23`), dipasangkan setelah klasterisasi (*post-hoc profiling*) dengan metadata harian: rata-rata `T2M`, total curah hujan `TQL`, hari dalam minggu, `holiday`, `school`, dan era (pra-COVID vs COVID-19).
* **Metrik Evaluasi:** **Silhouette Score**, **Davies-Bouldin Index**, **Calinski-Harabasz Index**, *Elbow Method (Inertia)*, serta analisis tabel kontingensi / **Adjusted Rand Index (ARI)** terhadap label `is_weekend` dan `holiday`.
* **Catatan Penerapan:** Pada matriks 2.002 hari, hari terjadinya *blackout* 20 Januari 2019 berpotensi membentuk klaster tunggal (singleton) atau memengaruhi centroid K-Means; bandingkan klasterisasi dengan *K-Medoids* atau keluarkan hari blackout saat pemetaan arketipe utama.

---

### Jalur 7: *Explainable AI* (XAI) & Interpretabilitas Faktor Penggerak Beban

* **Pertanyaan Riset:** Bagaimana kontribusi relatif fitur historis, meteorologi tiga kota, dan kalender terhadap prediksi beban di tingkat global maupun lokal (misalnya jam beban puncak siang hari vs dini hari, serta hari kerja normal vs lockdown April 2020)?
* **Metode / Model:**
  - **SHAP (*SHapley Additive exPlanations*):** Menggunakan **TreeSHAP** pada model LightGBM/XGBoost/CatBoost atau **DeepSHAP/KernelSHAP** pada model jaringan saraf.
  - **Partial Dependence Plot (PDP) & Individual Conditional Expectation (ICE):** Memetakan respons rata-rata prediksi terhadap perubahan `T2M_toc` atau `T2M_san` serta efek heterogen per jam.
  - **Permutation Feature Importance:** Mengukur penurunan performa (RMSE/MAE) ketika setiap variabel diacak secara independen pada data uji.
* **Feature Engineering & Variabel:** Kelompokkan fitur SHAP secara semantik (*grouped SHAP*) menjadi empat klaster penjelas: (1) *Autoregressive Lags*, (2) *Cuaca Tocumen/Santiago/David*, (3) *Kalender & Siklikal Waktu*, dan (4) *Tren Makro*.
* **Metrik Evaluasi:** Konsistensi peringkat fitur antar-metode (SHAP vs Permutation Importance), nilai mean $|SHAP|$ per kelompok fitur, serta rasio kontribusi fitur pada jam beban puncak (*peak hours*) vs luar jam puncak (*off-peak hours*).
* **Catatan Penerapan:** Sangat hati-hati saat menafsirkan SHAP pada fitur yang sangat berkorelasi (misalnya `T2M_toc`, `T2M_san`, dan `T2M_dav`), karena SHAP dapat membagi kredit antar fitur suhu yang kolinear; gunakan *SHAP feature clustering* atau analisis berbasis kelompok fitur (*grouped importance*).

---

## 5. Catatan Metodologis & Potensi *Pitfall* (Khusus Dataset Ini)

Agar hasil pemodelan valid secara statistik dan bebas dari kesalahan metodologis umum pada data deret waktu energi, perhatikan lima hal berikut:

1. **Validasi Temporal (Anti *Data Leakage*)**
   Jangan pernah menggunakan *random train-test split* atau *random K-Fold CV* pada data per jam. Observasi `t` dan `t+1` memiliki autokorelasi sangat tinggi sehingga pemisahan acak akan menghasilkan akurasi semu (*over-optimistic*). Gunakan:
   - *Chronological split* (mis. Latih: 2015–2018, Validasi: 2019, Uji Khusus: 2020; atau Latih: 2015–2018, Uji Normal: 2019), atau
   - *Expanding / Rolling Window Time-Series Cross-Validation* (`TimeSeriesSplit`).

2. **Pemisahan Rezim Normal vs COVID-19**
   Maret–Juni 2020 mengalami penurunan beban −12% hingga −17% YoY akibat *structural break* pandemi. Jika data uji mencakup 2020 tanpa mekanisme adaptasi (*online learning* / pembaruan bobot berkala) atau indikator perubahan rezim, metrik error akan melonjak drastis bukan karena model buruk di situasi normal, melainkan akibat *concept drift*. Selalu laporkan metrik evaluasi **terpisah** untuk periode normal (2019) dan periode pandemi (Mar–Jun 2020).

3. **Penanganan *Blackout* 20 Januari 2019 (dan Drop 2015/2017)**
   Pada `2019-01-20` pukul `12:00–16:00`, beban jatuh hingga `85,19 MWh` (jauh di bawah rata-rata `1.182,87 MWh`).
   - Pada **Jalur 1, 2, 4** (peramalan & GAM): titik ekstrem ini dapat mendistorsi fungsi rugi kuadratik (RMSE/MSE) serta merusak fitur *lag* pada 24–168 jam berikutnya. Lakukan imputasi/interpolasi musiman pada data latih (atau gunakan *robust loss* seperti Huber/MAE) dan laporkan evaluasi dengan serta tanpa titik *blackout* di data uji.
   - Pada **Jalur 5** (deteksi anomali): biarkan titik-titik tersebut utuh sebagai *ground-truth* evaluasi detektor.

4. **Multikolinearitas Cuaca 3 Kota & Representasi Fitur Siklikal**
   - Ketiga kota memiliki dinamika suhu yang searah (`T2M_toc` r = 0,6528; `T2M_dav` r = 0,6483; `T2M_san` r = 0,6270). Pada model linear/GAM/SHAP, gunakan regularisasi (Ridge/ElasticNet), reduksi dimensi (PCA), atau rata-rata tertimbang/seleksi satu stasiun representatif.
   - Jam (`0–23`), hari (`0–6`), dan hari dalam tahun (`1–365`) bersifat siklikal (jam 23 berdekatan dengan jam 0). Encode menggunakan pasangan $\sin(2\pi x / P)$ dan $\cos(2\pi x / P)$ terutama untuk model linear, SVR, KNN, dan neural network.
   - Verifikasi ulang interpretasi `QV2M_*` sebelum menulis pembahasan meteorologis, mengingat rentang nilainya (`0,0097–0,0227`) mencerminkan *specific humidity* (kg/kg) alih-alih kelembapan relatif (%).

5. **Disiplin Lag & Ketersediaan Fitur Saat Prediksi**
   Bedakan dengan tegas skenario operasional sebelum menyusun fitur *lag*:
   - **VSTLF (1 jam ke depan):** Lag `t-1`, `t-2`, `t-3`, dan cuaca `t-1` valid digunakan.
   - **STLF 24 jam atau 168 jam (1 minggu) ke depan:** Pada saat memprediksi `t+168`, nilai `nat_demand` pada `t+167` **belum tersedia**. Gunakan hanya lag $\ge 168$ (atau $\ge 24$ untuk *day-ahead*), strategi *direct multi-step* (satu model per horizon), atau *recursive forecasting* dengan nilai prakiraan sebelumnya.

---

## 6. Matriks Rekomendasi Pemilihan Rancangan (Untuk Tugas / Riset S2)

Tabel berikut memetakan fokus riset ke kombinasi jalur analisis yang paling efektif beserta perkiraan kompleksitas dan ekosistem pustaka yang direkomendasikan:

| Fokus / Tujuan Riset | Kombinasi Jalur Disarankan | Tingkat Kompleksitas | Ekosistem Paket Python / R Utama |
| :--- | :--- | :--- | :--- |
| **Prediktif & Benchmark Akurasi** (Fokus performa STLF/VSTLF) | **Jalur 1** (Baseline → ML → DL/Hibrida) + **Jalur 2** (Quantile / Conformal) | Menengah – Tinggi | `lightgbm`, `xgboost`, `catboost`, `statsmodels`, `pytorch` / `neuralforecast`, `mapie` |
| **Kausal & Evaluasi Kebijakan** (Fokus dampak lockdown COVID-19) | **Jalur 3** (Counterfactual + CausalImpact / ITS) + **Jalur 4** (GAM) | Menengah | `pygam` / R `mgcv`, `causalimpact` / `tfp-causalimpact`, `ruptures`, `statsmodels` |
| **Eksploratif & Pemahaman Pola** (Fokus dinamika kurva & gangguan) | **Jalur 6** (Clustering 24 jam / DTW / FPCA) + **Jalur 5** (Deteksi Blackout) | Rendah – Menengah | `scikit-learn`, `tslearn`, `scikit-fda`, `statsmodels` (STL/MSTL) |
| **Explainable AI & Riset Komprehensif S2** (*End-to-End* Siap Publikasi) | **Jalur 1** + **Jalur 3** (Evaluasi Normal vs Pandemi) + **Jalur 7** (SHAP & PDP) | Tinggi | `lightgbm`, `xgboost`, `shap`, `scikit-learn`, `pygam`, `matplotlib`/`seaborn` |

---

*Disusun berdasarkan inspeksi langsung terhadap `Panama_beban_listrik_per_jam_2015-2020_.xlsx` dan tinjauan literatur Short-Term Load Forecasting (2016–2025).*
