---
tags: [project/captura, prd, ai-service, fastapi, computer-vision]
date: 2026-06-02
status: living-document
parent: [[../PRD|Captura AI PRD]]
---

# Captura AI Service PRD

## 1. Ringkasan

AI Service Captura adalah microservice yang menganalisis foto street photography agar foto dapat dicari berdasarkan parameter unik seperti lokasi/waktu, plat nomor kendaraan, tipe kendaraan, dan ciri visual seperti pakaian, warna, style, atau objek tertentu.

AI Service mengacu pada [[../PRD|Captura AI PRD]] sebagai konteks produk dan berintegrasi dengan [[../backend/PRD|Backend PRD]] sebagai orchestrator data.

## 2. Tujuan AI Service

- Mengekstrak metadata visual dari foto.
- Membaca plat nomor kendaraan jika terlihat.
- Mengklasifikasikan tipe kendaraan.
- Menghasilkan tag visual dan embedding untuk semantic search.
- Mengambil metadata EXIF jika tersedia.
- Mengembalikan hasil analisis yang stabil, terstruktur, dan fault-tolerant.

## 3. Input dan Output

## 3.1 Input

AI Service menerima:

- Image URL.
- Optional image file.
- Moment ID dari backend.
- Optional metadata manual seperti lokasi, waktu, dan fotografer.
- Optional flags untuk re-analysis.

## 3.2 Output

AI Service mengembalikan:

- EXIF metadata.
- Vehicle detection result.
- License plate OCR candidates.
- Visual tags.
- CLIP embedding.
- Scene/style descriptors.
- Confidence score per task.
- Processing warnings/errors.

Output harus tetap dikembalikan walaupun salah satu tahap gagal.

## 4. Pipeline Requirements

## 4.1 EXIF Extraction

Tujuan:

- Mengambil metadata bawaan foto.

Data yang dicari:

- Captured timestamp.
- GPS latitude/longitude.
- Camera make/model.
- Lens metadata jika tersedia.

Catatan:

- Banyak foto social media kehilangan EXIF, jadi pipeline tidak boleh bergantung penuh pada EXIF.

## 4.2 Vehicle Detection

Tujuan:

- Mendeteksi tipe kendaraan yang muncul di foto.

Target awal:

- Motorcycle.
- Car.
- Bicycle.
- Scooter.
- Bus.
- Truck.
- Unknown.

Model saat ini:

- YOLOv8 atau model object detection sejenis.

Output:

- Vehicle type.
- Bounding boxes.
- Confidence.
- Dominant vehicle.

## 4.3 License Plate OCR

Tujuan:

- Membaca plat nomor kendaraan yang terlihat.

Kebutuhan:

- OCR kandidat plat.
- Normalisasi karakter.
- Regex/fallback untuk pola plat Indonesia.
- Confidence score.
- Partial plate support.

Catatan:

- OCR harus toleran terhadap blur, angle, overexposure, dan occlusion.
- Backend/frontend dapat melakukan fuzzy matching berdasarkan kandidat OCR.

## 4.4 Visual Tagging

Tujuan:

- Menghasilkan tag visual yang bisa dipakai untuk pencarian deskriptif.

Contoh tag:

- red jacket.
- white helmet.
- black motorcycle.
- runner.
- night street.
- golden hour.
- group ride.

Model:

- CLIP zero-shot classification atau prompt-based similarity.

## 4.5 Embedding Generation

Tujuan:

- Menghasilkan vector embedding untuk semantic search.

Kebutuhan:

- Embedding per image.
- Optional text embedding untuk query.
- Format konsisten agar bisa disimpan di backend pgvector atau index FAISS.

## 5. Search Support Requirements

AI Service harus mendukung backend untuk:

- Image-to-text metadata enrichment.
- Query-to-vector embedding.
- Similarity search experiment.
- Re-ranking candidate moments berdasarkan semantic similarity.

Untuk MVP, backend dapat menyimpan embedding dan menjalankan search, sementara AI service fokus pada analysis dan embedding generation.

## 6. API Requirements

## 6.1 Analyze Image

Endpoint:

- `POST /analyze`

Fungsi:

- Menerima image URL atau payload.
- Menjalankan pipeline EXIF, vehicle, plate, tags, embedding.
- Mengembalikan structured result.

## 6.2 Generate Text Embedding

Endpoint planned:

- `POST /embed/text`

Fungsi:

- Menerima query pengguna.
- Menghasilkan embedding untuk semantic search.

## 6.3 Health Check

Endpoint:

- `GET /health`

Fungsi:

- Memastikan service hidup.
- Optional model readiness status.

## 7. Output Schema Requirements

Contoh output konseptual:

```json
{
  "status": "completed",
  "exif": {
    "captured_at": "2026-05-16T17:00:00+07:00",
    "latitude": -6.9147,
    "longitude": 107.6098
  },
  "vehicle": {
    "dominant_type": "motorcycle",
    "confidence": 0.91
  },
  "plates": [
    {
      "text": "D 1234 AB",
      "normalized": "D1234AB",
      "confidence": 0.78
    }
  ],
  "tags": ["black motorcycle", "white helmet", "golden hour"],
  "embedding": [0.012, -0.034],
  "warnings": []
}
```

## 8. Non-Functional Requirements

## 8.1 Fault Tolerance

- Jika OCR gagal, vehicle detection tetap dikembalikan.
- Jika EXIF kosong, visual analysis tetap berjalan.
- Jika CLIP gagal, OCR dan YOLO tetap dikembalikan.

## 8.2 Performance

- Single image analysis MVP idealnya selesai dalam beberapa detik.
- Batch upload dapat diproses asynchronous.
- Model harus lazy-loaded atau warmed up untuk mengurangi cold-start penalty.

## 8.3 Accuracy

- Plate OCR harus menyimpan beberapa kandidat, bukan hanya satu hasil.
- Vehicle detection harus menyimpan confidence.
- Visual tags harus dapat dikurasi atau ditambah manual dari backend.

## 8.4 Privacy

- AI service tidak melakukan face recognition berbasis identitas personal untuk MVP.
- Metadata sensitif seperti plat nomor dikirim kembali ke backend sebagai internal metadata.

## 9. Integration with Backend

Backend bertanggung jawab:

- Menyimpan image dan metadata.
- Memanggil AI service.
- Menyimpan hasil analisis.
- Mengatur retry.
- Menentukan metadata publik vs internal.

AI Service bertanggung jawab:

- Analisis image.
- Embedding.
- Structured response.
- Model/runtime health.

## 10. MVP Acceptance Criteria

- AI service dapat menerima image URL.
- AI service dapat mengembalikan vehicle type.
- AI service dapat mengembalikan plate OCR candidates jika ada.
- AI service dapat mengembalikan visual tags.
- AI service dapat mengembalikan embedding.
- AI service tetap mengembalikan partial result ketika salah satu tahap gagal.
- Backend dapat menyimpan hasil output tanpa transformasi manual yang rumit.

## 11. Open Questions

- Apakah plate detection perlu model khusus sebelum OCR?
- Apakah CLIP cukup untuk style/pakaian, atau perlu model tambahan untuk apparel/person attributes?
- Apakah embedding search production akan memakai FAISS atau pgvector?
- Apakah batch processing perlu queue khusus dari backend?
- Bagaimana threshold confidence untuk menampilkan plate match di UI publik?

