# 📂 Dataset Guide: Lip Texture Condition Dataset

Dataset ini berisi foto/citra area bibir yang diklasifikasikan ke dalam 3 kategori kondisi kesehatan berdasarkan tekstur visual permukaan bibir:

1. **`sehat`**: Bibir dengan kondisi epitel sehat, terhidrasi, tanpa keretakan mendalam atau peradangan.
2. **`kurang_sehat`**: Bibir kering, dehidrasi ringan, deskuamasi (pengelupasan stratum korneum), atau pecah-pecah halus.
3. **`tidak_sehat`**: Bibir dengan fisur/luka dalam, keilitis, radang parah, atau pendarahan mikro.

---

## 🗂️ Struktur Direktori yang Diharapkan

Data dapat diletakkan dalam direktori `data/` dengan hierarki berikut:

```text
data/
└── processed/
    ├── train/
    │   ├── kurang_sehat/  (230 images)
    │   ├── sehat/         (230 images)
    │   └── tidak_sehat/   (230 images)
    ├── val/
    │   ├── kurang_sehat/  (100 images)
    │   ├── sehat/         (100 images)
    │   └── tidak_sehat/   (100 images)
    └── test/
        ├── kurang_sehat/  (100 images)
        ├── sehat/         (100 images)
        └── tidak_sehat/   (100 images)
```

---

## ⚙️ Pra-pemrosesan Data
- **Normalisasi Resolusi**: Citra di-resize ke $224 \times 224$ piksel.
- **Grayscale Conversion**: Citra 3-channel RGB dikonversi menjadi citra 1-channel grayscale untuk mengisolasi fitur tekstur dan kerutan epitel bibir serta mereduksi bias warna kulit/lipstik.
- **Undersampling**: Penyeimbangan jumlah sampel di setiap kelas untuk menghindari bias mayoritas saat training.
