# Project pengganti UAS Deep Learning
### Normalisasi Pewarnaan vs Augmentasi Warna pada Segmentasi Nukleus–Sitoplasma Sel Darah Putih Lintas Domain

**Tugas Besar Pengganti UAS · Sub-topik T3-A · Deep Learning for Image – Computer Vision Lanjutan (Kelas Deep Learning A)**
Program Studi S1 Sistem Informasi · Tim Tam · Dosen pengampu: Dr. Wahyudi Setiawan, S.Kom., M.Kom.

| Anggota | NIM | Peran utama |
|---|---|---|
| Elfina Lukna Zunaida (ketua) | 240441100013 | Data dan EDA, audit kebocoran, model U-Net, normalisasi pewarnaan (Ablasi 1), evaluasi |
| Nayzila Sukma Laudyandana | 240441100037 | Model DeepLabV3+, pelatihan multi-seed, color jitter (Ablasi 2), Grad-CAM dan galeri kesalahan, efisiensi, uji ketahanan warna |

---

## 1. Ringkasan

Model segmentasi sel darah putih (WBC) yang bagus di satu laboratorium sering menurun di laboratorium lain karena pewarnaan, pencahayaan, dan alat pemotretan berbeda (pergeseran domain). Ada dua cara umum mengatasinya:

1. **Normalisasi pewarnaan**: menyeragamkan warna citra sebelum masuk model (Reinhard, Macenko).
2. **Augmentasi warna**: melatih model dengan variasi warna (color jitter) agar tidak peka terhadap warna.

Proyek ini membandingkan keduanya secara terkontrol pada dataset Zheng dkk. (2018) yang kecil dan berbeda warna (D1 kekuningan, D2 CellaVision keunguan), memakai **U-Net** dan **DeepLabV3+** (ResNet-34, bobot ImageNet), dua arah domain (D1→D2 dan D2→D1), dan tiga seed.

**Pertanyaan riset.** Apakah normalisasi pewarnaan (Reinhard/Macenko) lebih efektif daripada color jitter dalam mengecilkan selisih Dice antara uji in-domain dan cross-domain?

**Temuan utama.** Color jitter kuat memberi selisih Dice sitoplasma (in-domain − cross-domain) terkecil pada keempat kombinasi model–arah, tanpa menambah waktu inferensi. Pada U-Net D1→D2 selisihnya 19,52 ± 1,29 poin, dibanding 33,23 ± 8,12 (baseline) dan 33,18 ± 8,33 (Macenko).

## 2. Hipotesis dan hasil

| Kode | Hipotesis | Hasil | Status |
|---|---|---|---|
| H1 | Jitter kuat memperkecil selisih Dice sitoplasma D1→D2 lebih besar daripada Macenko, minimal 2 poin (rerata 3 seed) | Macenko − jitter kuat = +13,66 poin (U-Net, 3 seed); pada seed 42 saja +4,36 | Terbukti |
| H2 | Reinhard dan Macenko setara (selisih Dice < 1 poin) | Selisih Dice (in − cross), seed 42: D1→D2 41,85 poin; D2→D1 1,05 poin | Tidak terbukti |
| H3 | Pada pergeseran warna buatan berat, penurunan Dice sitoplasma jitter lebih kecil daripada baseline dan normalisasi | Jitter 1,60 vs baseline 26,78 vs normalisasi 6,85 poin (U-Net D1→D2); terpenuhi pada 4 dari 4 kombinasi model × arah | Terbukti |

Hipotesis yang tidak terbukti tetap dilaporkan apa adanya.

### Selisih Dice sitoplasma (in-domain − cross-domain), rerata ± SD antar 3 seed (poin, makin kecil makin baik)

| Arah | Model | Baseline | Normalisasi (Macenko) | Jitter (kuat) |
|---|---|---|---|---|
| D1→D2 | U-Net | 33,23 ± 8,12 | 33,18 ± 8,33 | **19,52 ± 1,29** |
| D1→D2 | DeepLabV3+ | 31,66 ± 3,74 | 26,68 ± 2,26 | **20,25 ± 1,33** |
| D2→D1 | U-Net | 59,26 ± 3,99 | 30,10 ± 5,61 | **22,37 ± 9,00** |
| D2→D1 | DeepLabV3+ | 57,62 ± 4,46 | 22,53 ± 0,72 | **7,77 ± 1,41** |

Tabel lengkap (Dice dan IoU per kelas, mIoU, HD95, hasil in-domain) ada di `tables/lampiran_uji_utama_semua_metrik.csv`.

## 3. Data
Link Drive : https://drive.google.com/drive/folders/1QEl4H5icYMdaE0JIh8pKQJcQiB2rCqX8?usp=sharing

- **Sumber:** Zheng dkk. (2018), *Micron* 107, 55–71. Data: Mendeley Data, DOI [10.17632/w7cvnmn4c5.1](https://doi.org/10.17632/w7cvnmn4c5.1).
- **D1:** 300 citra mikroskop biasa (ukuran asli 88–120 piksel, 30 variasi). **D2:** 100 citra CellaVision (300×300). Masker 3 kelas: latar (0), sitoplasma (128), nukleus (255).
- **Dataset tidak diunggah ke repositori.** Unduh sendiri dari tautan di atas. Berkasnya berupa 4 arsip `.rar`: `Dataset 1`, `Dataset 2`, `Result 1`, `Result 2`.
- **Perhatian:** citra `.bmp` dan masker `.png` berada dalam **satu folder** (jadi folder tampak berisi 600 dan 200 berkas). Folder `Result 1/2` berisi masker biner hasil metode penulis asli dan **bukan ground truth**, sehingga tidak dipakai.
- **Hasil audit:** 300/300 dan 100/100 pasangan citra–masker lengkap; 5 masker D2 punya nilai di luar 3 label (034, 035, 047, 050, 076) dan dikuantisasi ke label terdekat; 1 pasang duplikat persis dan 3 pasang sangat mirip di dalam D1 (digabung dalam satu grup dan ditempatkan di subset yang sama); 0 duplikat di D2 dan antar D1–D2; kebocoran antar subset 0.
- **Pra-pemrosesan:** semua citra diubah ke 256×256 (citra bilinear, masker nearest); diperiksa visual pada 24 citra.
- **Split tetap:** seed 42, 70/10/20 per dataset (D1 210/30/60, D2 70/10/20), tersimpan di `splits/split_seed42.csv`.

## 4. Desain eksperimen

| Kelompok | Isi |
|---|---|
| Baseline | Tanpa normalisasi, tanpa jitter; hanya augmentasi geometrik (flip horizontal/vertikal, rotasi 90°) |
| Ablasi 1 (Elfina) | U-Net, seed 42, kedua arah: tanpa normalisasi vs Reinhard vs Macenko |
| Ablasi 2 (Nayzila) | U-Net, seed 42, kedua arah: jitter lemah vs sedang vs kuat |
| Uji utama | U-Net dan DeepLabV3+, kedua arah, baseline vs normalisasi terbaik vs jitter terbaik, seed 42/43/44 (36 run) |
| Uji ketahanan warna | Semua model uji utama, citra uji domain sumber diberi pergeseran warna buatan ringan/sedang/berat, hanya inferensi |
| Uji eksternal LISC | Opsional; **tidak dilakukan** (data LISC dengan masker nukleus–sitoplasma yang sesuai tidak tersedia) |

Total **42 run latih** (12 ablasi + 36 uji utama − 6 yang sama), tidak ada yang gagal.

**Normalisasi.** Reinhard (statistik Lab) dan Macenko (dua komponen pewarna) dipasang sebagai pra-pemrosesan pada semua citra. Acuan warna dihitung hanya dari set latih sumber. Bila Macenko gagal pada suatu citra, dipakai Reinhard (terjadi pada 0 citra).

**Color jitter** (torchvision `ColorJitter`, hanya saat pelatihan):

| Level | Kecerahan / kontras / saturasi | Hue |
|---|---|---|
| Lemah | 0,1 | 0,02 |
| Sedang | 0,3 | 0,05 |
| Kuat | 0,5 | 0,10 |

**Pergeseran warna buatan** (uji ketahanan; hue digeser ke dua arah lalu dirata-rata):

| Tingkat | Geser hue | Faktor saturasi | Faktor kecerahan |
|---|---|---|---|
| Ringan | ±0,03 | ×0,80 | ×0,85 |
| Sedang | ±0,08 | ×0,60 | ×0,70 |
| Berat | ±0,15 | ×0,40 | ×0,55 |

**Aturan pemilihan konfigurasi.** Normalisasi dan level jitter terbaik dipilih dari Dice sitoplasma pada **set validasi domain target** (rerata dua arah). Set uji tidak pernah dipakai untuk memilih. Terpilih: Macenko dan jitter kuat. Pemakaian label validasi target ini adalah keterbatasan.

## 5. Konfigurasi pelatihan (sama untuk semua run kecuali faktor yang diuji)

| Komponen | Nilai |
|---|---|
| Model | U-Net ResNet-34 dan DeepLabV3+ ResNet-34 (`segmentation_models_pytorch`), bobot ImageNet, fine-tuning penuh |
| Loss; optimizer | CE + Dice; Adam, lr 1e-4 |
| Masukan; batch | 256×256; 16 |
| Epoch | Maksimum 100, early stopping pada Dice validasi domain sumber (kesabaran 15 epoch), bobot terbaik dipulihkan |
| Presisi | Mixed precision (AMP) saat pelatihan dan prediksi; latensi diukur float32 |
| Penskalaan masukan | Rerata/simpangan baku ImageNet (bukan normalisasi pewarnaan) |
| Seed | Ablasi: 42. Uji utama: 42, 43, 44 |
| Perangkat | Tesla T4 (Google Colab) |
| Pustaka | PyTorch 2.14.1+cu130, segmentation_models_pytorch 0.6.0 (lengkap di `logs/lingkungan.json`) |
| Total waktu latih | ±1,02 jam GPU (42 run; belum termasuk evaluasi, Grad-CAM, uji ketahanan) |

## 6. Metrik

Dice dan IoU per citra untuk nukleus dan sitoplasma (rerata ± SD), mIoU (rerata IoU kedua kelas), HD95 (persentil ke-95 jarak Hausdorff, piksel pada 256×256), dan selisih in-domain − cross-domain. Satu skrip evaluasi dipakai untuk semua model.

Penanganan masker kosong: bila prediksi dan masker asli sama-sama kosong, Dice = IoU = 1 dan HD95 = 0; bila hanya salah satu kosong, Dice = IoU = 0 dan HD95 = diagonal citra (≈ 362 piksel). Pada tabel multi-seed, setiap seed menghasilkan satu nilai (rerata per citra), lalu dilaporkan rerata ± SD (n−1) antar tiga seed.

## 7. Efisiensi

| Model | Parameter (juta) | Latensi batch 1 (ms/citra) | Citra/detik |
|---|---|---|---|
| U-Net ResNet-34 | 24,44 | 7,8 ± 0,9 | 128,6 |
| DeepLabV3+ ResNet-34 | 22,44 | 7,5 ± 0,8 | 134,0 |

Diukur pada Tesla T4, float32, 10 citra warm-up, rerata 60 citra. Normalisasi menambah waktu pra-pemrosesan 24,7 ms/citra (Reinhard) dan 32,5 ms/citra (Macenko) di CPU; jitter tidak menambah waktu inferensi.


**Penamaan run:** `<arsitektur>_<sumber>to<target>_<kondisi>_s<seed>`, misalnya `unet_D1toD2_baseline_s42`. Arsitektur: `unet`, `deeplab`. Kondisi: `baseline`, `reinhard`, `macenko`, `jitter_lemah`, `jitter_sedang`, `jitter_kuat`.

**Tidak diunggah:** dataset Zheng dan folder `ckpt/` (bobot model, fp16). Keduanya dibuat ulang oleh notebook.

## 8. Cara mereproduksi

**Kebutuhan:** akun Google (Colab dan Drive), runtime GPU T4, dan sekitar 3–5 jam waktu total bila dijalankan penuh (boleh dicicil).

1. **Siapkan data.** Unduh dataset dari Mendeley Data (DOI di atas), lalu unggah ke `MyDrive/TimTam_T3A/data/`. Notebook membaca berkas dari `ZIP_PATH` (default `data/zheng.zip`); bila datanya berupa `.rar` terpisah atau sudah diekstrak, isi `RAW_OVERRIDE` atau letakkan berkasnya di folder `data_raw`.
2. **Buka notebook di Colab** dan pilih *Runtime → Change runtime type → T4 GPU*.
3. **Jalankan sel dari atas ke bawah.** Catatan per bagian:

| Bagian | Sel | Isi | Catatan |
|---|---|---|---|
| A. Setup | 1–6 | Pasang pustaka, konfigurasi, ekstrak data, pasangkan citra–masker | Sel 4 perlu `unrar` untuk `.rar`. Sel 5: `MANUAL` mengarah ke folder `Dataset 1` dan `Dataset 2` (citra dan masker satu folder). Sel 2b menghapus hasil lama, hanya sekali (penanda `logs/.reset_v2_done`). |
| B. Audit | 7–13 | Cek nilai masker, EDA, duplikat/kebocoran, split, cek visual | Menghasilkan `splits/` dan `logs/` audit |
| C. Alat bantu | 14–20 | Normalisasi, jitter, metrik, model, pelatihan, efisiensi, pelaksana run | Definisi fungsi, belum ada pelatihan |
| D. Baseline | 21–22 | Baseline M2 (U-Net, latih D1) | ±1,5 menit |
| E. Ablasi | 23–24 | 12 run ablasi dan pemilihan konfigurasi | Aman bila terputus: run selesai dilewati |
| F. Uji utama | 25–28 | 36 run (3 seed) dan Tabel 4, 5, Gambar 2 | Bagian terlama; boleh dicicil |
| G. Ketahanan warna | 29–30 | Pergeseran warna buatan, Tabel 6, Gambar 3 | Hanya inferensi |
| H. Analisis | 31–33 | Galeri kesalahan, Tabel 9, Grad-CAM | Deskriptif, bukan untuk memilih model |
| I. Ringkasan | 34–38 | Efisiensi, uji hipotesis, LISC (opsional), kemas hasil | Sel 36 otomatis dilewati bila LISC tidak diatur |

4. **Mode uji cepat:** set `QUICK_TEST = True` di Sel 2 untuk mencoba seluruh alur dengan 2 epoch (hasil di `results_quicktest`). Kembalikan ke `False` untuk eksperimen sungguhan.
5. **Hasil** tersimpan di Google Drive (`MyDrive/TimTam_T3A/`), sehingga aman bila Colab terputus.

**Catatan reproduksibilitas.** Seed diatur untuk Python, NumPy, dan PyTorch, tetapi hasil GPU tidak dijamin identik bit-per-bit antar sesi atau versi pustaka. Versi `grad-cam` dan `imagehash` tidak tercatat otomatis; ambil dengan `pip show grad-cam imagehash`.

## 9. Keterbatasan

- Data sangat kecil (D2 hanya 100 citra; set uji D2 hanya 20 citra), sehingga SD antar seed besar dan selisih kecil sebaiknya tidak ditafsirkan berlebihan.
- Hanya dua domain yang diuji, dan setiap citra berisi satu sel; hasil belum tentu berlaku untuk apusan padat atau sel bertumpuk.
- Label validasi domain target dipakai untuk memilih konfigurasi, yang tidak selalu tersedia pada kasus nyata.
- Ablasi dijalankan pada satu seed (42); hasil seed 42 bisa berbeda dari rerata tiga seed (mis. Macenko D1→D2: 68,50 vs 60,54).
- Pergeseran warna buatan hanya meniru sebagian perbedaan antar laboratorium, dan tingkat berat mendekati rentang variasi yang dilihat model jitter kuat saat pelatihan.
- Kategori kesalahan ditentukan dengan aturan otomatis (luas dan jumlah komponen); pelabelan manual belum dilakukan. Grad-CAM hanya dibuat untuk baseline dan jitter kuat (tidak untuk normalisasi) dan hanya dugaan, bukan bukti.
- Uji eksternal LISC tidak dilakukan.

## 10. Pernyataan penggunaan AI generatif

Dalam proses pengerjaan, tim memanfaatkan Claude (Anthropic) sebagai pendamping diskusi dan alat bantu pendukung, terutama untuk menyusun kode Colab, memeriksa ketepatan bahasa,Seluruh gagasan dan keputusan utama, mulai dari perumusan pertanyaan riset dan hipotesis, perancangan eksperimen, bersumber dari tim. Setiap masukan dari AI kami telaah dan sesuaikan terlebih dahulu sebelum digunakan

## 11. Referensi utama

- Zheng, X., Wang, Y., Wang, G., & Liu, J. (2018). Fast and robust segmentation of white blood cell images by self-supervised learning. *Micron*, 107, 55–71.
- Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. *MICCAI 2015*.
- Chen, L.-C., Zhu, Y., Papandreou, G., Schroff, F., & Adam, H. (2018). Encoder-decoder with atrous separable convolution for semantic image segmentation. *ECCV 2018*.
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *CVPR 2016*.
- Reinhard, E., Ashikhmin, M., Gooch, B., & Shirley, P. (2001). Color transfer between images. *IEEE CG&A*, 21(5), 34–41.
- Macenko, M., dkk. (2009). A method for normalizing histology slides for quantitative analysis. *IEEE ISBI 2009*.
- Tellez, D., dkk. (2019). Quantifying the effects of data augmentation and stain color normalization in convolutional neural networks for computational pathology. *Medical Image Analysis*, 58, 101544.
- Selvaraju, R. R., dkk. (2017). Grad-CAM: Visual explanations from deep networks via gradient-based localization. *ICCV 2017*.
- Iakubovskii, P. (2019). Segmentation Models PyTorch. https://github.com/qubvel/segmentation_models.pytorch
