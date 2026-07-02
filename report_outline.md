# Report Outline - Wake Word Detection Berbasis HMM-GMM

Gunakan file ini sebagai kerangka laporan akhir PDF.

## 1. Cover
Judul: Wake Word Detection Berbasis HMM-GMM untuk Wake Word "Hey, Jarvis"

## 2. Pendahuluan
Jelaskan latar belakang wake word detection, peran sistem berbasis suara, dan alasan menggunakan pendekatan klasik MFCC + HMM-GMM.

## 3. Deskripsi Wake Word
Wake word yang digunakan adalah "Hey, Jarvis".

Poin analisis:
- Frasa pendek dan mudah diucapkan.
- Memiliki dua bagian fonetik yang cukup jelas: "Hey" dan "Jarvis".
- Tidak terlalu mirip dengan kata umum dalam bahasa Indonesia sehari-hari.
- Potensi error dapat muncul jika ada kata mirip, noise, atau pengucapan terlalu pelan.

## 4. Dataset
Dataset terdiri dari:
- 1 jam audio wake word dengan minimal 100 kemunculan.
- 1 jam audio non-wake word tanpa wake word.
- File anotasi CSV berisi `filename`, `start_time`, `end_time`, `label`, `speaker`, dan `environment`.

## 5. Preprocessing
Tahapan:
- Konversi/standarisasi WAV 16 kHz mono 16-bit PCM.
- Normalisasi amplitudo.
- Voice Activity Detection sederhana menggunakan energy/silence trimming.
- Segmentasi wake word berdasarkan anotasi.
- Segmentasi non-wake dari audio non-wake.

## 6. Ekstraksi Fitur MFCC
Jelaskan framing, windowing, FFT, Mel filterbank, log energy, DCT, cepstral coefficients, delta, dan delta-delta.

Parameter:
- Sampling rate: 16 kHz
- Frame length: 25 ms
- Frame shift: 10 ms
- MFCC: 13
- Delta dan delta-delta: digunakan
- Total fitur: 39 dimensi

## 7. Desain Model HMM-GMM
Model:
- Wake-word HMM-GMM
- Non-wake HMM-GMM

Parameter default:
- State: 5
- Gaussian mixture per state: 4
- Covariance: diagonal
- Topologi: left-to-right/Bakis

## 8. Training Model
Dataset dibagi menjadi:
- Training: 70%
- Validation: 15%
- Testing: 15%

Threshold dipilih dari validation set berdasarkan F1 terbaik, kemudian FAR lebih rendah.

## 9. Evaluasi
Sajikan:
- Confusion matrix
- Accuracy
- Precision
- Recall
- F1-score
- FAR
- FRR

## 10. Analisis Hasil
Bahas pengaruh threshold terhadap false alarm dan missed detection.

## 11. Strengths and Weaknesses
Strengths:
- Tidak membutuhkan deep learning.
- Lebih ringan secara komputasi.
- Cocok untuk dataset kecil.
- Lebih mudah dijelaskan secara teoritis.

Weaknesses:
- Sensitif terhadap noise.
- Bergantung pada segmentasi dan threshold.
- Kurang fleksibel terhadap variasi suara ekstrem.
- Performa dapat turun jika kondisi testing berbeda dari training.

## 12. Kesimpulan
Rangkum performa sistem dan kelayakan HMM-GMM untuk wake word detection sederhana.

## 13. Referensi
- Materi COMP6822001 Speech Recognition Session 1-12.
- Jurafsky, D. & Martin, J. H. Speech and Language Processing.
- Dokumentasi librosa, hmmlearn, scikit-learn.

## 14. Lampiran Source Code
Cantumkan struktur folder, potongan kode utama, dan cara menjalankan program.
