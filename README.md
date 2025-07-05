# Analisis Sentimen Ulasan Aplikasi Tokopedia

## Deskripsi Proyek

Proyek ini melakukan analisis sentimen terhadap ulasan aplikasi Tokopedia yang diambil dari Google Play Store. Sistem ini mengklasifikasikan ulasan menjadi tiga kategori sentimen: **Positif**, **Negatif**, dan **Netral** menggunakan berbagai teknik machine learning dan deep learning.

## Fitur Utama

- **Web Scraping**: Mengumpulkan ulasan aplikasi Tokopedia dari Google Play Store
- **Text Preprocessing**: Pembersihan dan normalisasi teks bahasa Indonesia
- **Multiple Model Comparison**: Perbandingan performa berbagai algoritma klasifikasi
- **Interactive Prediction**: Sistem prediksi sentimen real-time
- **Comprehensive Evaluation**: Evaluasi model dengan berbagai metrik akurasi

## Teknologi yang Digunakan

### Library Utama
- **Data Processing & Visualization**
  - `pandas` - Manipulasi dan analisis data
  - `numpy` - Operasi numerik
  - `matplotlib` & `seaborn` - Visualisasi data

### Text Preprocessing (Bahasa Indonesia)
- **NLTK** - Natural Language Toolkit untuk tokenisasi
- **Sastrawi** - Library khusus bahasa Indonesia
  - `StemmerFactory` - Stemming kata bahasa Indonesia
  - `StopWordRemoverFactory` - Penghapusan kata tidak penting

### Machine Learning & NLP
- **Scikit-learn**
  - `TfidfVectorizer` - Ekstraksi fitur TF-IDF
  - `LabelEncoder` - Encoding label kategori
  - Multiple classifiers (SVM, Random Forest, Naive Bayes, dll.)

### Deep Learning
- **TensorFlow/Keras**
  - `Sequential` - Model neural network
  - `LSTM` - Long Short-Term Memory untuk sequence processing
  - `Embedding` - Word embedding layer

## Arsitektur Model

### 1. Preprocessing Pipeline
```python
def preprocess_text(text):
    # Pembersihan teks
    # Tokenisasi
    # Stopword removal
    # Stemming
    return processed_text
```

### 2. Feature Extraction
- **TF-IDF Vectorization**
  - Max features: 1000
  - Min document frequency: 5
  - Max document frequency: 0.7
  - N-gram range: (1,2)
  - Sublinear TF scaling

### 3. Model Comparison Results

| Model | Training Accuracy | Testing Accuracy | Status |
|-------|------------------|------------------|--------|
| **SVM (LinearSVC)** | **93.49%** | **91.61%** | ✅ **Selected** |
| Logistic Regression | 92.46% | 90.57% | ⚡ Good |
| Random Forest | 98.56% | 85.54% | ⚠️ Overfitting |
| Decision Tree | 99.18% | 80.95% | ⚠️ Overfitting |
| Naive Bayes | 75.95% | 73.93% | ❌ Underperforming |
| LSTM Deep Learning | 96.82% | 92.09% | ⚡ Good |

## Model Terpilih: Support Vector Machine (SVM)

### Alasan Pemilihan SVM:
1. **Balanced Performance**: Akurasi training (93.49%) dan testing (91.61%) yang seimbang
2. **No Overfitting**: Gap antara training dan testing accuracy hanya ~2%
3. **Robust**: Performa konsisten pada data yang tidak terlihat
4. **Efficient**: Waktu training yang relatif cepat
5. **Interpretable**: Hasil prediksi dapat dijelaskan dengan baik

### Konfigurasi Model Terpilih:
```python
svm_model = LinearSVC(random_state=42)
tfidf = TfidfVectorizer(
    max_features=1000,
    min_df=5,
    max_df=0.7,
    ngram_range=(1,2),
    sublinear_tf=True
)
```

## Instalasi

### Requirements
```bash
pip install pandas numpy matplotlib seaborn
pip install nltk scikit-learn
pip install Sastrawi
pip install tensorflow
pip install requests
```

### Setup NLTK Data
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## Penggunaan

### 1. Training Model
```python
# Load dan preprocess data
X = clean_df['text_akhir']
y = clean_df['sentimen_lexicon']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature extraction
tfidf = TfidfVectorizer(max_features=1000, min_df=5, max_df=0.7, ngram_range=(1,2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Training
svm_model = LinearSVC(random_state=42)
svm_model.fit(X_train_tfidf, y_train)
```

### 2. Prediksi Sentimen
```python
def predict_sentiment(text):
    processed_text = preprocess_text(text)
    text_tfidf = tfidf.transform([processed_text])
    numeric_pred = svm_model.predict(text_tfidf)[0]
    sentiment = le.inverse_transform([numeric_pred])[0]
    return sentiment, processed_text
```

### 3. Sistem Interactive
```python
manual_input_system()
```

## Contoh Hasil Prediksi

| Input | Prediksi |
|-------|----------|
| "Aplikasi sangat user friendly, proses belanja jadi lebih cepat dan mudah!" | **POSITIF** |
| "Pengiriman super cepat, barang sampai sebelum estimasi. Seller ramah-ramah" | **POSITIF** |
| "Aplikasi sering error pas checkout, bikin frustrasi!" | **NEGATIF** |
| "Barang tidak sesuai deskripsi, seller sulit dihubungi" | **NEGATIF** |
| "Layanan standar saja" | **NETRAL** |

## Struktur Proyek

```
Analisis-Sentimen-Ulasan-Aplikasi-Tokopedia/
├── analisis_sentimen_Tokopedia.ipynb    # Notebook utama untuk analisis sentimen
├── requirements.txt                      # Daftar dependencies Python
├── scrapping_data.py                    # Script untuk scraping data dari Play Store
├── ulasan_tokopedia.csv                 # Dataset ulasan Tokopedia
└── README.md                            # Dokumentasi proyek
```

## Evaluasi Model

### Metrik Evaluasi:
- **Accuracy**: 91.61%
- **Precision**: Diukur per kelas sentimen
- **Recall**: Diukur per kelas sentimen
- **F1-Score**: Harmonic mean precision dan recall

### Confusion Matrix:
Model SVM menunjukkan performa yang baik dalam membedakan ketiga kelas sentimen dengan tingkat kesalahan klasifikasi yang rendah.

## Pengembangan Selanjutnya

1. **Data Augmentation**: Menambah variasi data training
2. **Hyperparameter Tuning**: Optimasi parameter model
3. **Ensemble Methods**: Kombinasi multiple models
4. **Real-time Monitoring**: Tracking performa model secara real-time
5. **API Development**: Membuat REST API untuk integrasi
6. **Web Interface**: Dashboard untuk analisis sentimen

## Kontribusi

Kontribusi sangat diterima! Silakan fork repository ini dan submit pull request untuk perbaikan atau penambahan fitur.


---

**Catatan**: Proyek ini dikembangkan untuk keperluan edukasi dan penelitian. Pastikan untuk mematuhi terms of service dari Google Play Store saat melakukan scraping data.
