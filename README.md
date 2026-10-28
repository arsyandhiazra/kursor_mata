# Kursor Mata

Menggerakkan kursor mouse dengan arah pandangan mata, memakai kamera web biasa. Tanpa perangkat khusus.

Penulis: **Zulfa Arsyandhi Azra**

## Fitur

- Kursor mengikuti arah pandangan mata (MediaPipe Face Landmarker, termasuk titik iris).
- Klik dengan kedip sengaja: kiri, ganda, dan kanan.
- Pelatihan (kalibrasi) pertama kali, lalu diuji pada titik-titik baru supaya akurasinya terukur.
- Hasil pelatihan disimpan, jadi pembukaan berikutnya langsung siap pakai.
- Data latih bisa ditumpuk antarsesi agar makin akurat.
- Penyaring bergaya fiksasi: kursor stabil saat menatap, dan cepat menyusul saat tatapan pindah.

## Isi repositori

| Berkas | Fungsi |
| --- | --- |
| `kursor_mata.py` | Program utama |
| `Kursor_Mata.command` | Peluncur klik ganda untuk macOS |
| `face_landmarker.task` | Model pelacak wajah dari Google MediaPipe |
| `requirements.txt` | Daftar paket Python |
| `index.html` | Halaman web proyek (untuk GitHub Pages) |

## Kebutuhan

- Python 3
- Kamera web
- macOS (paling cocok, ada peluncurnya). Di Windows atau Linux program seharusnya bisa dijalankan langsung dengan `python kursor_mata.py`, tetapi belum diuji.

## Pemasangan

```bash
git clone https://github.com/<nama-pengguna-anda>/kursor-mata.git
cd kursor-mata
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Peluncur `Kursor_Mata.command` otomatis memakai `.venv` bila ada.

## Menjalankan

```bash
python3 kursor_mata.py
```

Atau, di macOS, klik ganda `Kursor_Mata.command`. Bila tidak mau terbuka:

```bash
chmod +x Kursor_Mata.command
```

Pertama kali, klik kanan berkas itu lalu pilih **Buka** agar lolos pemeriksaan Gatekeeper.

Izinkan juga kamera dan **Aksesibilitas** untuk Terminal di Pengaturan Sistem > Privasi & Keamanan. Tanpa izin Aksesibilitas, kursor tidak akan bergerak atau mengeklik.

## Pelatihan pertama

Duduk tegak sekitar 50 sampai 60 cm dari layar dan jaga kepala tetap diam; hanya mata yang bergerak. Program menampilkan 20 titik (5 kolom x 4 baris) sebanyak 3 putaran, lalu menguji akurasi pada 7 titik baru. Hasilnya disimpan di `~/.kursor_mata`.

## Kontrol

| Gerakan | Hasil |
| --- | --- |
| Kedip sekali | Klik kiri |
| Kedip dua kali berurutan | Klik ganda |
| Kedip lama (sekitar 0,7 detik atau lebih) | Klik kanan |
| Mata tertutup 2 detik | Jeda atau lanjut |

Kedip alami yang sangat cepat diabaikan.

Tombol di jendela pratinjau: `q` atau `Esc` keluar, `p` jeda, `c` klik hidup/mati, `t` tambah data latih, `k` latih ulang dari nol.

## Opsi baris perintah

| Opsi | Keterangan |
| --- | --- |
| `--kamera N` | Nomor kamera (bawaan 0) |
| `--kalibrasi-ulang` | Latih ulang walau sudah ada hasil tersimpan |
| `--tambah` | Latih lagi dan tumpuk dengan sesi sebelumnya |
| `--putaran N` | Jumlah putaran pelatihan (bawaan 3, maksimal 6) |
| `--tanpa-uji` | Lewati uji akurasi |
| `--halus X` | Pengali radius fiksasi: kecil = lebih cepat menyusul, besar = lebih stabil |
| `--tanpa-klik` | Matikan klik dengan kedip |
| `--kedip-min`, `--kedip-maks`, `--kedip-lama`, `--jendela-ganda` | Atur durasi kedip (detik) |

## Tips akurasi

- Cahaya terang merata di wajah, tanpa cahaya kuat dari belakang.
- Kamera sejajar dengan mata.
- Bila galat uji masih besar, jalankan `--kalibrasi-ulang --putaran 4`, atau tekan `t` beberapa kali untuk menumpuk data.

## Privasi

Gambar kamera diproses di komputer Anda dan tidak disimpan. Yang tersimpan di `~/.kursor_mata` hanya angka fitur mata dan hasil pelatihan. Koneksi internet hanya dipakai bila `face_landmarker.task` tidak ditemukan dan perlu diunduh dari Google.

## Halaman web (GitHub Pages)

Berkas `index.html` adalah halaman proyek. Untuk membukanya sebagai situs:

1. Buka **Settings > Pages** di repositori GitHub.
2. Pada **Build and deployment**, pilih **Deploy from a branch**.
3. Pilih branch `main` dan folder `/ (root)`, lalu **Save**.
4. Tunggu satu sampai dua menit. Situs terbuka di `https://<nama-pengguna-anda>.github.io/kursor-mata/`.

Halaman ini hanya menjelaskan proyek dan memuat demo penyaring pandangan. Programnya sendiri tetap berjalan di komputer Anda, karena browser tidak boleh menggerakkan kursor sistem.

## Kredit

- Dibuat oleh Zulfa Arsyandhi Azra.
- Pelacak wajah: [MediaPipe Face Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/face_landmarker) dari Google.

## Lisensi

[MIT](LICENSE) © 2026 Zulfa Arsyandhi Azra. Model `face_landmarker.task` tetap tunduk pada ketentuan resmi Google MediaPipe.
