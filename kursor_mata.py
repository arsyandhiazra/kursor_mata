#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kursor Mata: menggerakkan kursor mouse dengan arah pandangan mata
memakai kamera web biasa.

Pertama kali dijalankan, program melatih (kalibrasi) pemetaan pandangan ke layar,
menguji akurasinya pada titik-titik baru, lalu menyimpannya di folder ~/.kursor_mata.
Pada pembukaan berikutnya program langsung memakai hasil pelatihan itu.

Kontrol klik (dengan kedip sengaja):
- Kedip sekali              : klik kiri
- Kedip dua kali berurutan  : klik kiri dua kali (klik ganda)
- Kedip lama                : klik kanan
- Mata tertutup 2 detik     : jeda / lanjut
Kedip alami yang sangat cepat diabaikan, jadi tidak memicu klik.

Jendela pratinjau: q atau Esc = keluar, p = jeda, c = klik hidup/mati,
t = tambah data pelatihan (menumpuk dengan sesi sebelumnya),
k = latih ulang dari nol.

Menumpuk hasil latihan: data mentah tiap sesi disimpan di ~/.kursor_mata/data_latih_v3.npz.
Dengan --tambah (atau tombol t), sesi baru digabung dengan sesi-sesi sebelumnya
(maksimal 8 sesi terbaru; sesi lebih lama diberi bobot lebih kecil), lalu model dilatih ulang
dari seluruh data gabungan.

Penulis: Zulfa Arsyandhi Azra
"""
import os
import sys

# Bisukan log C++ bawaan MediaPipe/TFLite sebelum pustakanya dimuat
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import argparse
import math
import shutil
import ssl
import time
import urllib.request

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
from mediapipe.tasks import python as tugas_mp
from mediapipe.tasks.python import vision

pyautogui.FAILSAFE = False  # kursor boleh menyentuh sudut layar
pyautogui.PAUSE = 0

# ---------------------------------------------------------------- pengaturan
VERSI = 3                   # naik bila format fitur/berkas berubah; berkas versi lain diabaikan
FOLDER_SKRIP = os.path.dirname(os.path.abspath(__file__))
FOLDER_DATA = os.path.join(os.path.expanduser("~"), ".kursor_mata")
BERKAS_KALIBRASI = os.path.join(FOLDER_DATA, "kalibrasi_mata_v3.npz")
BERKAS_DATA = os.path.join(FOLDER_DATA, "data_latih_v3.npz")   # data mentah semua sesi pelatihan
NAMA_MODEL = "face_landmarker.task"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/face_landmarker/"
             "face_landmarker/float16/1/face_landmarker.task")

# Pelatihan: kisi titik dan jumlah putaran (setiap putaran mengunjungi semua titik)
KOLOM, BARIS = 5, 4
MARGIN = 0.06
PUTARAN_BAWAAN = 3
JEDA_SEBELUM_SAMPEL = 0.8   # detik, memberi waktu mata berpindah ke titik
DURASI_SAMPEL = 1.3         # detik, lama pengambilan data per titik
MIN_SAMPEL_TITIK = 12       # sampel bersih minimal per titik; kurang dari ini titik diulang
MAKS_ULANG = 2              # berapa kali satu titik boleh diulang
MAKS_SESI = 8               # sesi pelatihan terbaru yang disimpan saat menumpuk data
PELURUHAN_SESI = 0.8        # bobot sesi lama dikalikan 0.8 untuk tiap sesi yang lebih tua

# Titik uji akurasi (pecahan layar). Sengaja di antara titik kisi latihan,
# supaya galat yang terukur adalah galat nyata pada posisi yang belum pernah dilatih.
TITIK_UJI = [(0.39, 0.20), (0.61, 0.20), (0.17, 0.50), (0.50, 0.50),
             (0.83, 0.50), (0.39, 0.80), (0.61, 0.80)]
BATAS_GALAT_UJI = 0.12      # pecahan tinggi layar; di atas ini program menyarankan latih ulang

# Klik dengan kedip (detik)
KEDIP_MIN_BAWAAN = 0.20     # di bawah ini dianggap kedip alami, diabaikan
KEDIP_MAKS_BAWAAN = 0.45    # kedip sengaja pendek
KEDIP_LAMA_BAWAAN = 0.70    # mulai dari sini dianggap kedip lama (klik kanan)
JENDELA_GANDA_BAWAAN = 0.50 # batas waktu kedip kedua agar menjadi klik ganda
DURASI_JEDA_PROGRAM = 2.0

# Keterbukaan mata relatif terhadap keadaan terbuka normal (ref_buka)
AMBANG_TUTUP = 0.55         # di bawah ini mata dianggap tertutup
AMBANG_BUKA = 0.70          # di atas ini mata dianggap terbuka lagi
AMBANG_PRA = 0.85           # di bawah ini kursor sudah ditahan (kelopak mulai menutup)

# Penyaring pandangan
FIKSASI_N_MAKS = 12         # rata-rata berjalan paling banyak 12 bingkai saat menatap
FIKSASI_KONFIRMASI = 3      # bingkai berturut-turut di luar radius agar dianggap pindah tatapan
JEDA_MATI_PIKSEL = 4        # kursor tidak digeser bila perubahan lebih kecil dari ini
MARGIN_Z = 0.75             # batas ekstrapolasi fitur di luar rentang data latih (satuan sd)

# Nomor titik pada model Face Landmarker (478 titik, termasuk iris)
MATA_A = dict(sudut_a=33, sudut_b=133, atas=159, bawah=145, iris=468)
MATA_B = dict(sudut_a=362, sudut_b=263, atas=386, bawah=374, iris=473)

# Vektor fitur mentah (11 kolom):
#   0 h, 1 v_kelopak, 2 v_sudut      : posisi iris dalam mata
#   3 bs_h, 4 bs_v                   : arah lirik dari blendshape model
#   5 kepala_x, 6 kepala_y, 7 putar  : arah hadap kepala (dari matriks pose)
#   8 tx, 9 ty, 10 tz                : posisi kepala terhadap kamera
JUMLAH_FITUR = 11
KUMPULAN_FITUR = {
    "iris": [0, 1, 2],
    "iris+blend": [0, 1, 2, 3, 4],
    "iris+posisi": [0, 1, 2, 8, 9, 10],
    "iris+kepala": [0, 1, 2, 5, 6, 7, 8, 9, 10],
    "penuh": list(range(11)),
}
NILAI_LAM = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)

FONT = cv2.FONT_HERSHEY_SIMPLEX
_waktu_terakhir_ms = 0
_peringatan_versi = False


class KameraMati(Exception):
    pass


def bisukan_log_native():
    """
    Mengalihkan log C++ (peringatan MediaPipe/TFLite yang tampak seperti galat) ke /dev/null,
    tetapi pesan galat Python tetap tampil di terminal.
    """
    try:
        sys.stderr.flush()
        asli = os.dup(2)
        gelap = os.open(os.devnull, os.O_WRONLY)
        os.dup2(gelap, 2)
        os.close(gelap)
        sys.stderr = os.fdopen(asli, "w", buffering=1)
    except OSError:
        pass


def periksa_izin_mac():
    if sys.platform != "darwin":
        return
    try:
        from ApplicationServices import AXIsProcessTrusted
        if not AXIsProcessTrusted():
            print("PERINGATAN: Terminal belum punya izin Aksesibilitas, jadi kursor tidak akan "
                  "bergerak atau mengeklik.\n"
                  "  Buka Pengaturan Sistem > Privasi & Keamanan > Aksesibilitas, aktifkan Terminal, "
                  "lalu jalankan ulang.")
    except Exception:
        pass   # modul tidak ada: lewati pemeriksaan


def baca(kamera, coba=40):
    """Membaca satu bingkai; bila kamera terus gagal, berhenti dengan pesan jelas."""
    for _ in range(coba):
        ok, frame = kamera.read()
        if ok and frame is not None:
            return frame
        time.sleep(0.01)
    raise KameraMati("Kamera tidak memberi gambar. Periksa izin kamera atau sambungan kamera, "
                     "lalu jalankan ulang.")


def tutup_jendela(nama):
    try:
        cv2.destroyWindow(nama)
    except cv2.error:
        pass
    for _ in range(4):   # macOS baru benar-benar menutup jendela setelah waitKey
        cv2.waitKey(1)


# ---------------------------------------------------------------- model wajah
def stempel_waktu():
    """Cap waktu dalam milidetik; harus selalu naik untuk mode video."""
    global _waktu_terakhir_ms
    t = int(time.monotonic() * 1000)
    if t <= _waktu_terakhir_ms:
        t = _waktu_terakhir_ms + 1
    _waktu_terakhir_ms = t
    return t


def cari_model():
    """Mencari berkas model; bila belum ada, mengunduhnya sekali ke folder data."""
    os.makedirs(FOLDER_DATA, exist_ok=True)
    tujuan = os.path.join(FOLDER_DATA, NAMA_MODEL)
    if os.path.exists(tujuan):
        return tujuan

    di_skrip = os.path.join(FOLDER_SKRIP, NAMA_MODEL)
    if os.path.exists(di_skrip):
        try:
            shutil.copy(di_skrip, tujuan)  # salin supaya aplikasi tetap jalan bila skrip dipindah
            return tujuan
        except OSError:
            return di_skrip

    print("Mengunduh model pelacak wajah (sekali saja, sekitar 4 megabita)...")
    try:
        try:
            import certifi
            konteks = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            konteks = ssl.create_default_context()
        sementara = tujuan + ".unduh"
        with urllib.request.urlopen(MODEL_URL, context=konteks, timeout=60) as r, \
                open(sementara, "wb") as f:
            f.write(r.read())
        os.replace(sementara, tujuan)
    except Exception as galat:
        raise SystemExit(
            f"Gagal mengunduh model ({galat}).\n"
            f"Unduh manual lewat terminal:\n"
            f"  curl -L -o \"{tujuan}\" {MODEL_URL}"
        )
    return tujuan


def buat_landmarker():
    opsi = vision.FaceLandmarkerOptions(
        base_options=tugas_mp.BaseOptions(model_asset_path=cari_model()),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
    )
    return vision.FaceLandmarker.create_from_options(opsi)


# ---------------------------------------------------------------- fitur mata
def titik(lm, i, lebar_gambar, tinggi_gambar):
    p = lm[i]
    return np.array([p.x * lebar_gambar, p.y * tinggi_gambar])


def fitur_mata(lm, idx, lebar_gambar, tinggi_gambar):
    a = titik(lm, idx["sudut_a"], lebar_gambar, tinggi_gambar)
    b = titik(lm, idx["sudut_b"], lebar_gambar, tinggi_gambar)
    atas = titik(lm, idx["atas"], lebar_gambar, tinggi_gambar)
    bawah = titik(lm, idx["bawah"], lebar_gambar, tinggi_gambar)
    iris = titik(lm, idx["iris"], lebar_gambar, tinggi_gambar)

    ab = b - a
    lebar_mata = np.linalg.norm(ab) + 1e-6
    horizontal = np.dot(iris - a, ab) / lebar_mata ** 2

    tinggi_mata = np.linalg.norm(bawah - atas) + 1e-6
    vertikal_kelopak = np.dot(iris - atas, bawah - atas) / tinggi_mata ** 2

    normal = np.array([-ab[1], ab[0]]) / lebar_mata
    vertikal_sudut = np.dot(iris - (a + b) / 2, normal) / lebar_mata

    keterbukaan = tinggi_mata / lebar_mata
    return horizontal, vertikal_kelopak, vertikal_sudut, keterbukaan


def pose_kepala(matriks):
    """Arah hadap dan posisi kepala dari matriks transformasi 4x4. None bila matriks tak valid."""
    M = np.asarray(matriks, dtype=float)
    if M.shape != (4, 4) or not np.all(np.isfinite(M)):
        return None
    R = M[:3, :3]
    sx, sz = np.linalg.norm(R[:, 0]), np.linalg.norm(R[:, 2])
    if sx < 1e-6 or sz < 1e-6:
        return None
    kanan, depan = R[:, 0] / sx, R[:, 2] / sz
    t = M[:3, 3]
    if np.linalg.norm(t) < 1e-6:   # susunan matriks transpos
        t = M[3, :3]
    return [depan[0], depan[1], kanan[1], t[0], t[1], t[2]]


def lirik_blendshape(hasil):
    """Arah lirik mata dari skor blendshape (horizontal, vertikal); 0 bila tak tersedia."""
    if not hasil.face_blendshapes:
        return 0.0, 0.0
    s = {c.category_name: c.score for c in hasil.face_blendshapes[0]}

    def g(nama):
        return float(s.get(nama, 0.0))

    h = ((g("eyeLookOutLeft") + g("eyeLookInRight"))
         - (g("eyeLookInLeft") + g("eyeLookOutRight"))) / 2
    v = ((g("eyeLookUpLeft") + g("eyeLookUpRight"))
         - (g("eyeLookDownLeft") + g("eyeLookDownRight"))) / 2
    return h, v


def ekstrak(frame_bgr, pelacak):
    """Mengembalikan (vektor_fitur[11], keterbukaan_mata) atau None bila wajah tak terdeteksi."""
    global _peringatan_versi
    tinggi, lebar = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    gambar = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    hasil = pelacak.detect_for_video(gambar, stempel_waktu())
    if not hasil.face_landmarks:
        return None
    if not hasil.facial_transformation_matrixes:
        if not _peringatan_versi:
            print("Versi mediapipe terlalu lama (tidak ada matriks pose kepala). "
                  "Jalankan: pip3 install -U mediapipe")
            _peringatan_versi = True
        return None
    pose = pose_kepala(hasil.facial_transformation_matrixes[0])
    if pose is None:
        return None

    lm = hasil.face_landmarks[0]
    f1 = fitur_mata(lm, MATA_A, lebar, tinggi)
    f2 = fitur_mata(lm, MATA_B, lebar, tinggi)
    bs_h, bs_v = lirik_blendshape(hasil)
    fitur = np.array([
        (f1[0] + f2[0]) / 2,   # horizontal
        (f1[1] + f2[1]) / 2,   # vertikal terhadap kelopak
        (f1[2] + f2[2]) / 2,   # vertikal terhadap sudut mata
        bs_h, bs_v,
        *pose,
    ], dtype=float)
    if not np.all(np.isfinite(fitur)):
        return None
    keterbukaan = (f1[3] + f2[3]) / 2
    return fitur, keterbukaan


# ---------------------------------------------------------------- pelatihan
def desain(F, konf):
    """Kolom fitur terpilih. Orde 'kuadrat' menambah suku kuadrat dan interaksi iris."""
    F = np.atleast_2d(F)
    kolom = F[:, KUMPULAN_FITUR[konf["kumpulan"]]]
    if konf["orde"] == "kuadrat":
        h, v = F[:, 0], F[:, 1]
        kolom = np.column_stack([kolom, h * v, h ** 2, v ** 2])
    return kolom


def tambah_bias(Z):
    return np.column_stack([np.ones(len(Z)), Z])


def cocokkan(Z, Y, bobot, lam):
    """Regresi ridge terbobot."""
    X = tambah_bias(Z)
    reg = lam * bobot.sum() * np.eye(X.shape[1])
    reg[0, 0] = 0  # konstanta tidak diregularisasi
    A = X.T @ (X * bobot[:, None]) + reg
    B = X.T @ (Y * bobot[:, None])
    return np.linalg.solve(A, B)


def cocokkan_kokoh(Z, Y, bobot, lam, iterasi=6):
    """Regresi ridge dengan pembobotan Huber berulang agar tahan terhadap data pencilan."""
    W = cocokkan(Z, Y, bobot, lam)
    X = tambah_bias(Z)
    for _ in range(iterasi):
        sisa = np.linalg.norm(X @ W - Y, axis=1)
        sigma = 1.4826 * np.median(sisa) + 1e-6
        batas = max(1.345 * sigma, 15.0)
        bobot_huber = np.where(sisa <= batas, 1.0, batas / np.maximum(sisa, 1e-9))
        W = cocokkan(Z, Y, bobot * bobot_huber, lam)
    return W


def galat_silang(Z, Y, bobot, daftar_kelompok, lam):
    """
    Validasi silang ketat: tiap kali satu kelompok disisihkan penuh sebagai data uji.
    Kelompok pertama = titik layar (galat pada posisi yang belum pernah dilihat model);
    kelompok kedua = putaran (galat antarwaktu). Hasilnya rata-rata keduanya.
    """
    per_skema = []
    for grup in daftar_kelompok:
        hasil = []
        for g in np.unique(grup):
            uji = grup == g
            if uji.all():
                continue
            W = cocokkan_kokoh(Z[~uji], Y[~uji], bobot[~uji], lam, iterasi=3)
            sisa = np.linalg.norm(tambah_bias(Z[uji]) @ W - Y[uji], axis=1)
            hasil.append(np.average(sisa, weights=bobot[uji]))
        if hasil:
            per_skema.append(np.mean(hasil))
    return float(np.mean(per_skema)) if per_skema else float("inf")


def bersihkan(f, buka, dasar=None):
    """
    Membuang sampel saat kelopak menutup sebagian, lalu pencilan.
    Batas keterbukaan relatif terhadap kunjungan itu sendiri (melihat ke bawah memang
    mempersempit mata). Mengembalikan None bila sampel bersih terlalu sedikit.
    """
    if len(f) == 0:
        return None
    batas = 0.88 * np.median(buka)
    if dasar is not None:
        batas = max(batas, 0.6 * dasar)
    f = f[buka >= batas]
    if len(f) < MIN_SAMPEL_TITIK:
        return None
    med = np.median(f[:, :3], axis=0)
    mad = 1.4826 * np.median(np.abs(f[:, :3] - med), axis=0) + 0.005
    f = f[np.all(np.abs(f[:, :3] - med) <= 3 * mad, axis=1)]
    return f if len(f) >= MIN_SAMPEL_TITIK else None


def latih(kunjungan, konfigurasi=None):
    """
    kunjungan: daftar dict {fitur (n x 11), buka (n), target (x, y), putaran, sesi}.
    "putaran" adalah nomor kelompok validasi silang (unik untuk tiap putaran di tiap sesi);
    "sesi" adalah nomor sesi pelatihan. Sesi yang lebih lama diberi bobot lebih kecil.
    konfigurasi (kumpulan, orde, lam) bila diberikan: lewati pencarian, hanya cocokkan ulang.
    Mengembalikan dict model atau None bila data tidak cukup.
    """
    if sum(len(k["buka"]) for k in kunjungan) < 150:
        print("Data pelatihan terlalu sedikit (wajah sering tidak terdeteksi). "
              "Perbaiki pencahayaan dan posisi duduk, lalu coba lagi.")
        return None

    sesi_terbaru = max(k["sesi"] for k in kunjungan)
    # Keterbukaan mata dasar dihitung per sesi, karena cahaya dan posisi duduk bisa berbeda
    dasar_sesi = {}
    for s in {k["sesi"] for k in kunjungan}:
        isi = [k["buka"] for k in kunjungan if k["sesi"] == s and len(k["buka"])]
        if isi:
            dasar_sesi[s] = float(np.median(np.concatenate(isi)))
    if sesi_terbaru not in dasar_sesi:
        print("Sesi pelatihan terbaru tidak punya data yang terbaca. Coba ulangi.")
        return None
    dasar = dasar_sesi[sesi_terbaru]

    F, Y, G_putaran, G_titik, B, panjang = [], [], [], [], [], []
    kunci_titik = {}
    for k in kunjungan:
        f = bersihkan(k["fitur"], k["buka"], dasar_sesi.get(k["sesi"], dasar))
        if f is None:
            continue
        kunci = (round(k["target"][0] / 8), round(k["target"][1] / 8))
        id_titik = kunci_titik.setdefault(kunci, len(kunci_titik))
        F.append(f)
        Y.append(np.tile(k["target"], (len(f), 1)))
        G_putaran.append(np.full(len(f), k["putaran"]))
        G_titik.append(np.full(len(f), id_titik))
        usia = PELURUHAN_SESI ** (sesi_terbaru - k["sesi"])
        B.append(np.full(len(f), usia / len(f)))        # tiap titik berbobot sama, sesi lama lebih ringan
        panjang.append(len(f))

    if len(F) < 8:
        print("Terlalu banyak titik yang gagal terbaca. Coba ulangi pelatihan.")
        return None

    F, Y = np.vstack(F), np.vstack(Y)
    G_putaran, G_titik = np.concatenate(G_putaran), np.concatenate(G_titik)
    B = np.concatenate(B)
    B *= len(B) / B.sum()

    daftar_kelompok = [G_titik]
    if len(np.unique(G_putaran)) >= 2:
        daftar_kelompok.append(G_putaran)

    if konfigurasi is None:
        calon = [(a, b, lam) for a in KUMPULAN_FITUR for b in ("linear", "kuadrat") for lam in NILAI_LAM]
    else:
        calon = [(konfigurasi["kumpulan"], konfigurasi["orde"], konfigurasi["lam"])]

    terbaik = None
    cache = {}
    for kumpulan, orde, lam in calon:
        if (kumpulan, orde) not in cache:
            mentah = desain(F, dict(kumpulan=kumpulan, orde=orde))
            mu = mentah.mean(axis=0)
            sd = mentah.std(axis=0)
            sd = np.where(sd < 1e-9, 1.0, sd)
            cache[(kumpulan, orde)] = (mu, sd, (mentah - mu) / sd)
        mu, sd, Z = cache[(kumpulan, orde)]
        try:
            g = galat_silang(Z, Y, B, daftar_kelompok, lam)
        except np.linalg.LinAlgError:
            continue
        if terbaik is None or g < terbaik["galat_validasi"]:
            terbaik = dict(kumpulan=kumpulan, orde=orde, lam=lam, mu=mu, sd=sd, Z=Z,
                           galat_validasi=g)
    if terbaik is None:
        print("Pencocokan model gagal (data tidak stabil). Coba ulangi pelatihan.")
        return None

    Z = terbaik["Z"]
    W = cocokkan_kokoh(Z, Y, B, terbaik["lam"])
    pred = tambah_bias(Z) @ W
    galat_latih = float(np.average(np.linalg.norm(pred - Y, axis=1), weights=B))

    # Derau per bingkai: sebaran prediksi di sekitar rata-rata pada tiap kunjungan
    derau, awal = [], 0
    for n in panjang:
        p = pred[awal:awal + n]
        derau.append(np.sqrt(np.mean(np.sum((p - p.mean(axis=0)) ** 2, axis=1))))
        awal += n
    derau = float(np.median(derau))

    banyak_sesi = len(dasar_sesi)
    print(f"Pelatihan selesai: {len(F)} sampel dari {len(panjang)} kunjungan titik "
          f"({banyak_sesi} sesi). Fitur '{terbaik['kumpulan']}', orde '{terbaik['orde']}', "
          f"galat latih {galat_latih:.0f} piksel, galat validasi silang "
          f"{terbaik['galat_validasi']:.0f} piksel, derau per bingkai {derau:.0f} piksel.")

    return dict(W=W, mu=terbaik["mu"], sd=terbaik["sd"], kumpulan=terbaik["kumpulan"],
                orde=terbaik["orde"], lam=terbaik["lam"], dasar=dasar,
                galat=terbaik["galat_validasi"], galat_latih=galat_latih, derau=derau,
                zmin=np.percentile(Z, 1, axis=0), zmax=np.percentile(Z, 99, axis=0),
                galat_uji=float("nan"))


def prediksi_banyak(model, F):
    """Prediksi untuk banyak bingkai. Mengembalikan (posisi N x 2, di_luar_jangkauan N)."""
    Z = (desain(F, model) - model["mu"]) / model["sd"]
    Zc = np.clip(Z, model["zmin"] - MARGIN_Z, model["zmax"] + MARGIN_Z)
    di_luar = np.any(np.abs(Z - Zc) > 2.0, axis=1)
    return tambah_bias(Zc) @ model["W"], di_luar


def prediksi(model, fitur):
    pos, luar = prediksi_banyak(model, fitur)
    return pos[0], bool(luar[0])


def simpan_model(model, lebar_layar, tinggi_layar):
    try:
        os.makedirs(FOLDER_DATA, exist_ok=True)
        np.savez(
            BERKAS_KALIBRASI, versi=VERSI, W=model["W"], mu=model["mu"], sd=model["sd"],
            kumpulan=np.array(model["kumpulan"]), orde=np.array(model["orde"]),
            lam=model["lam"], dasar=model["dasar"],
            galat_validasi=model["galat"], galat_latih=model["galat_latih"],
            derau=model["derau"], zmin=model["zmin"], zmax=model["zmax"],
            galat_uji=model["galat_uji"],
            lebar_layar=lebar_layar, tinggi_layar=tinggi_layar,
        )
        print(f"Hasil pelatihan disimpan di {BERKAS_KALIBRASI}")
    except OSError as galat:
        print(f"Peringatan: hasil pelatihan tidak bisa disimpan ({galat}). Program tetap berjalan.")


def muat_model(lebar_layar, tinggi_layar):
    if not os.path.exists(BERKAS_KALIBRASI):
        return None
    try:
        d = np.load(BERKAS_KALIBRASI)
        if int(d["versi"]) != VERSI:
            print("Hasil pelatihan tersimpan berformat lama. Melatih ulang.")
            return None
        W = d["W"].astype(float)
        mu, sd = d["mu"].astype(float), d["sd"].astype(float)
        if W.shape[0] != len(mu) + 1 or W.shape[1] != 2:
            raise ValueError("ukuran model tidak cocok")
        # Bila resolusi layar berbeda dari saat pelatihan, skalakan keluaran ke layar sekarang
        W = W * np.array([lebar_layar / float(d["lebar_layar"]),
                          tinggi_layar / float(d["tinggi_layar"])])
        return dict(W=W, mu=mu, sd=sd, kumpulan=str(d["kumpulan"]), orde=str(d["orde"]),
                    lam=float(d["lam"]), dasar=float(d["dasar"]),
                    galat=float(d["galat_validasi"]), galat_latih=float(d["galat_latih"]),
                    derau=float(d["derau"]), zmin=d["zmin"].astype(float),
                    zmax=d["zmax"].astype(float), galat_uji=float(d["galat_uji"]))
    except Exception as galat:
        print(f"Hasil pelatihan tersimpan tidak bisa dibaca ({galat}). Melatih ulang.")
        return None


def simpan_data(kunjungan, lebar_layar, tinggi_layar):
    """Menyimpan data mentah semua sesi supaya sesi berikutnya bisa ditumpuk."""
    try:
        os.makedirs(FOLDER_DATA, exist_ok=True)
        np.savez(
            BERKAS_DATA, versi=VERSI,
            fitur=np.vstack([k["fitur"].reshape(-1, JUMLAH_FITUR) for k in kunjungan]),
            buka=np.concatenate([k["buka"] for k in kunjungan]),
            id_kunjungan=np.concatenate(
                [np.full(len(k["buka"]), i) for i, k in enumerate(kunjungan)]).astype(int),
            tx=np.array([k["target"][0] / lebar_layar for k in kunjungan]),  # disimpan sebagai pecahan layar
            ty=np.array([k["target"][1] / tinggi_layar for k in kunjungan]),
            kelompok=np.array([k["putaran"] for k in kunjungan], dtype=int),
            sesi=np.array([k["sesi"] for k in kunjungan], dtype=int),
        )
    except OSError as galat:
        print(f"Peringatan: data pelatihan tidak bisa disimpan ({galat}).")


def muat_data(lebar_layar, tinggi_layar):
    """Memuat data mentah sesi-sesi sebelumnya; daftar kosong bila belum ada."""
    if not os.path.exists(BERKAS_DATA):
        return []
    try:
        d = np.load(BERKAS_DATA)
        if int(d["versi"]) != VERSI:
            print("Data pelatihan sebelumnya berformat lama. Dimulai dari nol.")
            return []
        fitur, buka, id_k = d["fitur"], d["buka"], d["id_kunjungan"]
        tx, ty, kelompok, sesi = d["tx"], d["ty"], d["kelompok"], d["sesi"]
        kunjungan = []
        for i in range(len(kelompok)):
            pilih = id_k == i
            kunjungan.append(dict(
                fitur=fitur[pilih], buka=buka[pilih],
                target=(float(tx[i]) * lebar_layar, float(ty[i]) * tinggi_layar),
                putaran=int(kelompok[i]), sesi=int(sesi[i]),
            ))
        return kunjungan
    except Exception as galat:
        print(f"Data pelatihan sebelumnya tidak bisa dibaca ({galat}). Dimulai dari nol.")
        return []


# ---------------------------------------------------------------- kalibrasi
def teks_tengah(kanvas, kalimat, y, skala=0.9, warna=(220, 220, 220)):
    ukuran = cv2.getTextSize(kalimat, FONT, skala, 2)[0]
    cv2.putText(kanvas, kalimat, ((kanvas.shape[1] - ukuran[0]) // 2, y), FONT, skala, warna, 2)


def tampil_pesan(nama, lebar_layar, tinggi_layar, kalimat):
    kanvas = np.zeros((tinggi_layar, lebar_layar, 3), np.uint8)
    teks_tengah(kanvas, kalimat, tinggi_layar // 2, 1.0, (0, 255, 255))
    cv2.imshow(nama, kanvas)
    cv2.waitKey(1)


def hitung_mundur(kamera, pelacak, nama, lebar_layar, tinggi_layar, detik=5):
    """Hitung mundur sambil memeriksa wajah terdeteksi. False bila dibatalkan (Esc)."""
    mulai = time.time()
    while time.time() - mulai < detik:
        ada = ekstrak(baca(kamera), pelacak) is not None
        sisa = detik - int(time.time() - mulai)
        kanvas = np.zeros((tinggi_layar, lebar_layar, 3), np.uint8)
        teks_tengah(kanvas, "Duduk tegak, wajah menghadap kamera, jarak sekitar 50-60 cm.",
                    tinggi_layar // 2 - 120)
        teks_tengah(kanvas, "Tatap setiap titik tanpa menggerakkan kepala.", tinggi_layar // 2 - 80)
        teks_tengah(kanvas, "Jangan menunduk atau memiringkan kepala; hanya mata yang bergerak.",
                    tinggi_layar // 2 - 40)
        teks_tengah(kanvas, f"Mulai dalam {sisa} detik", tinggi_layar // 2 + 30, 1.2, (0, 255, 255))
        teks_tengah(kanvas, "wajah terdeteksi" if ada else "wajah belum terdeteksi",
                    tinggi_layar // 2 + 90, 0.8, (0, 200, 0) if ada else (0, 0, 255))
        cv2.imshow(nama, kanvas)
        if cv2.waitKey(1) & 0xFF == 27:
            return False
    return True


def ambil_titik(kamera, pelacak, nama, lebar_layar, tinggi_layar, tx, ty, judul, catatan=""):
    """
    Menampilkan satu titik dan merekam fitur mata selama menatapnya.
    Mengembalikan (fitur n x 11, buka n) atau None bila dibatalkan (Esc).
    """
    total = JEDA_SEBELUM_SAMPEL + DURASI_SAMPEL
    fitur_k, buka_k = [], []
    mulai = time.time()
    while True:
        berlalu = time.time() - mulai
        if berlalu > total:
            break
        frame = baca(kamera)
        if berlalu >= JEDA_SEBELUM_SAMPEL:
            hasil = ekstrak(frame, pelacak)
            if hasil is not None:
                fitur_k.append(hasil[0])
                buka_k.append(hasil[1])

        kanvas = np.zeros((tinggi_layar, lebar_layar, 3), np.uint8)
        radius = int(28 - 18 * min(berlalu / total, 1.0))
        cv2.circle(kanvas, (int(tx), int(ty)), radius, (0, 255, 255), -1)
        cv2.circle(kanvas, (int(tx), int(ty)), 4, (0, 0, 0), -1)
        cv2.putText(kanvas, f"{judul}   (Esc = batal)", (30, 40), FONT, 0.7, (200, 200, 200), 2)
        if catatan:
            cv2.putText(kanvas, catatan, (30, 75), FONT, 0.7, (0, 0, 255), 2)
        cv2.imshow(nama, kanvas)
        if cv2.waitKey(1) & 0xFF == 27:
            return None
    return np.array(fitur_k).reshape(-1, JUMLAH_FITUR), np.array(buka_k)


def kunjungi_titik(kamera, pelacak, nama, lebar_layar, tinggi_layar, tx, ty, judul):
    """Seperti ambil_titik, tetapi mengulang titik bila datanya kurang bersih."""
    catatan = ""
    for percobaan in range(MAKS_ULANG + 1):
        hasil = ambil_titik(kamera, pelacak, nama, lebar_layar, tinggi_layar, tx, ty, judul, catatan)
        if hasil is None:
            return None
        if bersihkan(hasil[0], hasil[1]) is not None:
            return hasil
        catatan = "Data kurang bersih (wajah hilang / mata berkedip / bergerak). Mengulang titik."
    return hasil   # tetap dikembalikan; pelatihan akan membuangnya bila memang buruk


def evaluasi(model, kunjungan_uji):
    """Galat nyata (piksel) pada titik yang belum dilatih. Mengembalikan (galat_titik, derau)."""
    galat, derau = [], []
    for k in kunjungan_uji:
        f = bersihkan(k["fitur"], k["buka"], model["dasar"])
        if f is None:
            continue
        pred, _ = prediksi_banyak(model, f)
        galat.append(float(np.linalg.norm(np.median(pred, axis=0) - np.array(k["target"]))))
        derau.append(float(np.sqrt(np.mean(np.sum((pred - pred.mean(axis=0)) ** 2, axis=1)))))
    return galat, derau


def kalibrasi(kamera, pelacak, lebar_layar, tinggi_layar, putaran, tambah=False, uji=True):
    """
    Melatih pemetaan pandangan ke layar. Bila tambah=True, data sesi-sesi sebelumnya
    yang tersimpan ikut dipakai (ditumpuk); bila False, semua dimulai dari nol.
    Bila uji=True, akurasi diukur pada titik baru lalu titik itu ikut dilatihkan.
    """
    nama = "Kalibrasi"
    cv2.namedWindow(nama, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(nama, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    try:
        return _jalankan_kalibrasi(kamera, pelacak, nama, lebar_layar, tinggi_layar,
                                   putaran, tambah, uji)
    finally:
        tutup_jendela(nama)


def _jalankan_kalibrasi(kamera, pelacak, nama, lebar_layar, tinggi_layar, putaran, tambah, uji):
    lama = muat_data(lebar_layar, tinggi_layar) if tambah else []
    if tambah and not lama:
        print("Belum ada data sesi sebelumnya yang tersimpan; sesi ini menjadi sesi pertama.")
    sesi_baru = max((k["sesi"] for k in lama), default=-1) + 1

    if not hitung_mundur(kamera, pelacak, nama, lebar_layar, tinggi_layar):
        return None

    # Urutan titik berbentuk ular; putaran berikutnya dibalik agar urutannya berbeda
    dasar_urutan = []
    for i, y in enumerate(np.linspace(MARGIN, 1 - MARGIN, BARIS)):
        xs = np.linspace(MARGIN, 1 - MARGIN, KOLOM)
        if i % 2 == 1:
            xs = xs[::-1]
        for x in xs:
            dasar_urutan.append((x * lebar_layar, y * tinggi_layar))

    baru = []
    for r in range(putaran):
        urutan = dasar_urutan if r % 2 == 0 else dasar_urutan[::-1]
        for nomor, (tx, ty) in enumerate(urutan, 1):
            judul = f"Putaran {r + 1}/{putaran} - titik {nomor}/{len(urutan)}"
            hasil = kunjungi_titik(kamera, pelacak, nama, lebar_layar, tinggi_layar, tx, ty, judul)
            if hasil is None:
                return None
            baru.append(dict(fitur=hasil[0], buka=hasil[1], target=(tx, ty),
                             putaran=sesi_baru * 100 + r, sesi=sesi_baru))

    semua = lama + baru
    sesi_simpan = sorted({k["sesi"] for k in semua})[-MAKS_SESI:]   # simpan sesi terbaru saja
    semua = [k for k in semua if k["sesi"] in sesi_simpan]
    if lama:
        print(f"Menumpuk sesi baru dengan {len(sesi_simpan) - 1} sesi sebelumnya.")

    tampil_pesan(nama, lebar_layar, tinggi_layar, "Menghitung model...")
    model = latih(semua)
    if model is None:   # bila gagal, hasil dan data lama dibiarkan utuh
        return None

    if uji:
        if not hitung_mundur(kamera, pelacak, nama, lebar_layar, tinggi_layar, detik=3):
            print("Uji akurasi dilewati.")
        else:
            data_uji, batal = [], False
            for nomor, (px, py) in enumerate(TITIK_UJI, 1):
                tx, ty = px * lebar_layar, py * tinggi_layar
                judul = f"Uji akurasi - titik {nomor}/{len(TITIK_UJI)}"
                hasil = kunjungi_titik(kamera, pelacak, nama, lebar_layar, tinggi_layar, tx, ty, judul)
                if hasil is None:
                    batal = True
                    break
                data_uji.append(dict(fitur=hasil[0], buka=hasil[1], target=(tx, ty),
                                     putaran=sesi_baru * 100 + 90, sesi=sesi_baru))
            if batal:
                print("Uji akurasi dilewati.")
            else:
                laporan_uji(model, data_uji, tinggi_layar)
                semua = semua + data_uji   # titik uji menambah cakupan di antara titik kisi
                tampil_pesan(nama, lebar_layar, tinggi_layar, "Menghitung model akhir...")
                konf = dict(kumpulan=model["kumpulan"], orde=model["orde"], lam=model["lam"])
                akhir = latih(semua, konfigurasi=konf)
                if akhir is not None:
                    akhir["galat_uji"] = model["galat_uji"]
                    model = akhir

    simpan_model(model, lebar_layar, tinggi_layar)
    simpan_data(semua, lebar_layar, tinggi_layar)
    return model


def laporan_uji(model, data_uji, tinggi_layar):
    galat, derau = evaluasi(model, data_uji)
    if len(galat) < 3:
        print("Uji akurasi gagal: terlalu sedikit titik uji yang terbaca.")
        return
    rata, median, terburuk = float(np.mean(galat)), float(np.median(galat)), float(np.max(galat))
    model["galat_uji"] = rata
    print(f"Uji akurasi pada {len(galat)} titik baru: galat rata-rata {rata:.0f} piksel, "
          f"median {median:.0f}, terburuk {terburuk:.0f} "
          f"({100 * rata / tinggi_layar:.0f}% tinggi layar). "
          f"Derau per bingkai {float(np.median(derau)):.0f} piksel.")
    if rata > BATAS_GALAT_UJI * tinggi_layar:
        print("Galat uji masih besar. Perbaiki cahaya (wajah terang merata, tanpa cahaya dari belakang), "
              "kamera sejajar mata, jarak 50-60 cm, kepala diam, lalu jalankan dengan --kalibrasi-ulang "
              "--putaran 4, atau tekan t beberapa kali untuk menumpuk data.")


# ---------------------------------------------------------------- penyaring pandangan
class SaringanFiksasi:
    """
    Penyaring bergaya fiksasi/sakade. Selama mata menatap satu area, kursor memakai rata-rata
    berjalan (sangat stabil). Bila beberapa bingkai berturut-turut jatuh jauh dari area itu
    dan saling berdekatan, itu dianggap pindah tatapan: kursor langsung menyusul tanpa
    ketertinggalan seperti penyaring rata-rata biasa. Bingkai pencilan tunggal diabaikan.
    """

    def __init__(self, radius, n_maks=FIKSASI_N_MAKS, konfirmasi=FIKSASI_KONFIRMASI, lunak=0.85):
        self.radius = float(radius)
        self.n_maks = n_maks
        self.konfirmasi = konfirmasi
        self.lunak = lunak          # 1.0 = loncat langsung; lebih kecil = meluncur halus
        self.reset()

    def reset(self):
        self.pusat = None
        self.n = 0
        self.calon = []
        self.keluar = None

    def batalkan_calon(self):
        self.calon = []

    def __call__(self, p):
        p = np.asarray(p, dtype=float)
        if self.pusat is None:
            self.pusat, self.n, self.keluar = p.copy(), 1, p.copy()
            return self.keluar
        if np.linalg.norm(p - self.pusat) <= self.radius:
            self.n = min(self.n + 1, self.n_maks)
            self.pusat = self.pusat + (p - self.pusat) / self.n
            self.calon = []
        else:
            if self.calon and np.linalg.norm(p - self.calon[-1]) > 1.5 * self.radius:
                self.calon = []     # tidak konsisten: pencilan acak, mulai hitung ulang
            self.calon.append(p)
            if len(self.calon) >= self.konfirmasi:
                self.pusat = np.mean(self.calon, axis=0)
                self.n = len(self.calon)
                self.calon = []
        self.keluar = self.keluar + self.lunak * (self.pusat - self.keluar)
        return self.keluar


def hitung_radius(model, lebar_layar, tinggi_layar, faktor=1.0):
    """Radius fiksasi menyesuaikan derau terukur: makin berisik data, makin lebar tapi tetap dibatasi."""
    derau = model.get("derau", 30.0)
    if not np.isfinite(derau):
        derau = 30.0
    return float(np.clip(2.2 * derau * faktor, 30.0, 0.12 * min(lebar_layar, tinggi_layar)))


# ---------------------------------------------------------------- klik dengan kedip
class PengendaliKlik:
    def __init__(self, kedip_min, kedip_maks, kedip_lama, jendela_ganda):
        self.kedip_min = kedip_min
        self.kedip_maks = kedip_maks
        self.kedip_lama = kedip_lama
        self.jendela = jendela_ganda
        self.pertama_selesai = None   # waktu kedip pendek pertama selesai (menunggu kedip kedua)
        self.aksi = ""
        self.aksi_waktu = 0.0
        self.posisi = None            # posisi kursor yang dituju klik (x, y); None = posisi sekarang
        self.fungsi = dict(kiri=pyautogui.click, kanan=pyautogui.rightClick,
                           ganda=pyautogui.doubleClick)

    @property
    def menunggu(self):
        return self.pertama_selesai is not None

    def _lakukan(self, nama, jenis, sekarang):
        x, y = self.posisi if self.posisi is not None else (None, None)
        self.fungsi[jenis](x=x, y=y)   # klik tepat di posisi tatapan sebelum kedip
        self.aksi, self.aksi_waktu = nama, sekarang

    def kedip_selesai(self, mulai, selesai, aktif):
        """Dipanggil saat mata terbuka kembali. Mengembalikan 'jeda' untuk mengalihkan jeda."""
        durasi = selesai - mulai
        if durasi >= DURASI_JEDA_PROGRAM:
            self.pertama_selesai = None
            return "jeda"
        if not aktif:
            self.pertama_selesai = None
            return None

        # Kedip pertama yang menunggu dan sudah kedaluwarsa: selesaikan sebagai klik tunggal
        if self.menunggu and mulai - self.pertama_selesai > self.jendela:
            self._lakukan("klik kiri", "kiri", selesai)
            self.pertama_selesai = None

        if durasi >= self.kedip_lama:
            if self.menunggu:
                self._lakukan("klik kiri", "kiri", selesai)
                self.pertama_selesai = None
            self._lakukan("klik kanan", "kanan", selesai)
        elif self.kedip_min <= durasi <= self.kedip_maks:
            if self.menunggu:
                self._lakukan("klik ganda", "ganda", selesai)
                self.pertama_selesai = None
            else:
                self.pertama_selesai = selesai
        return None

    def perbarui(self, sekarang, sedang_tertutup):
        """Dipanggil tiap bingkai: bila kedip kedua tidak datang, lakukan klik tunggal."""
        if (self.menunggu and not sedang_tertutup
                and sekarang - self.pertama_selesai > self.jendela):
            self._lakukan("klik kiri", "kiri", sekarang)
            self.pertama_selesai = None

    def batal(self):
        self.pertama_selesai = None


# ---------------------------------------------------------------- program utama
def jalankan(args, kamera, lebar_layar, tinggi_layar, putaran):
    for _ in range(10):   # buang bingkai awal: eksposur kamera belum stabil
        kamera.read()

    with buat_landmarker() as pelacak:
        model = None
        if not (args.kalibrasi_ulang or args.tambah):
            model = muat_model(lebar_layar, tinggi_layar)
            if model is not None:
                uji = model["galat_uji"]
                teks_uji = f", galat uji {uji:.0f} piksel" if np.isfinite(uji) else ""
                print(f"Hasil pelatihan tersimpan dipakai (galat validasi silang sekitar "
                      f"{model['galat']:.0f} piksel{teks_uji}). Tekan t untuk menambah data "
                      f"pelatihan, k untuk melatih ulang dari nol.")
        if model is None:
            model = kalibrasi(kamera, pelacak, lebar_layar, tinggi_layar, putaran,
                              tambah=args.tambah, uji=not args.tanpa_uji)
            if model is None:
                raise SystemExit("Pelatihan dibatalkan atau gagal.")

        saringan = SaringanFiksasi(hitung_radius(model, lebar_layar, tinggi_layar, args.halus))
        klik = PengendaliKlik(args.kedip_min, args.kedip_maks, args.kedip_lama, args.jendela_ganda)
        ref_buka = model["dasar"]   # keterbukaan mata terbuka normal; menyesuaikan diri perlahan
        terakhir = None             # posisi kursor terakhir yang dikirim
        tertutup = False
        tutup_sejak = None
        pra = False                 # kelopak mulai menutup: kursor ditahan
        pra_sejak = 0.0
        tahan_sampai = 0.0          # kursor ditahan sebentar setelah mata terbuka kembali
        dijeda = False
        klik_aktif = not args.tanpa_klik
        luar = False

        print("Berjalan. Tekan q atau Esc pada jendela pratinjau untuk keluar.")
        while True:
            frame = baca(kamera)
            sekarang = time.time()
            status = "wajah tidak terdeteksi"
            rel = 0.0

            hasil = ekstrak(frame, pelacak)
            if hasil is not None:
                fitur, buka = hasil
                dasar = model["dasar"]
                rel = buka / ref_buka

                if not tertutup and rel < AMBANG_TUTUP:
                    tertutup, tutup_sejak, pra = True, sekarang, False
                    saringan.batalkan_calon()
                elif tertutup and rel > AMBANG_BUKA:
                    tertutup = False
                    aksi = klik.kedip_selesai(tutup_sejak, sekarang, klik_aktif and not dijeda)
                    if aksi == "jeda":
                        dijeda = not dijeda
                    tutup_sejak = None
                    tahan_sampai = sekarang + 0.20

                if not tertutup:
                    if rel < AMBANG_PRA:
                        if not pra:
                            pra, pra_sejak = True, sekarang
                        elif sekarang - pra_sejak > 0.5:
                            # Terlalu lama untuk kedip: kelopak memang lebih sempit (mis. melihat
                            # ke bawah). Pakai keterbukaan ini sebagai acuan baru.
                            ref_buka = float(np.clip(buka / 0.95, 0.5 * dasar, 1.3 * dasar))
                            pra = False
                    else:
                        pra = False
                        ref_buka = float(np.clip(ref_buka + 0.03 * (buka - ref_buka),
                                                 0.5 * dasar, 1.3 * dasar))

                if tertutup:
                    status = "mata tertutup"
                elif dijeda:
                    status = "dijeda"
                elif pra or klik.menunggu or sekarang < tahan_sampai:
                    status = "kursor ditahan"
                else:
                    pred, luar = prediksi(model, fitur)
                    pred = np.clip(pred, [1, 1], [lebar_layar - 2, tinggi_layar - 2])
                    posisi = saringan(pred)
                    x, y = int(round(posisi[0])), int(round(posisi[1]))
                    if terakhir is None or math.hypot(x - terakhir[0], y - terakhir[1]) > JEDA_MATI_PIKSEL:
                        pyautogui.moveTo(x, y)
                        terakhir = (x, y)
                    status = "aktif"
            else:
                pra = False
                if tertutup and sekarang - tutup_sejak > 1.5:
                    tertutup = False   # wajah hilang terlalu lama: batalkan tanpa aksi
                    tutup_sejak = None
                    klik.batal()

            klik.posisi = terakhir
            klik.perbarui(sekarang, tertutup)

            tampil = cv2.flip(cv2.resize(frame, (480, 270)), 1)
            cv2.putText(tampil, f"{status} | klik {'hidup' if klik_aktif else 'mati'}",
                        (10, 25), FONT, 0.6, (0, 255, 0), 2)
            if sekarang - klik.aksi_waktu < 1.0:
                cv2.putText(tampil, klik.aksi, (10, 58), FONT, 0.9, (0, 255, 255), 2)
            if luar and status == "aktif":
                cv2.putText(tampil, "kepala jauh dari posisi latih", (10, 255), FONT, 0.55,
                            (0, 140, 255), 2)
            cv2.imshow("Pratinjau Kursor Mata", tampil)

            tombol = cv2.waitKey(1) & 0xFF
            if tombol in (27, ord("q")):
                break
            elif tombol == ord("p"):
                dijeda = not dijeda
            elif tombol == ord("c"):
                klik_aktif = not klik_aktif
                klik.batal()
            elif tombol in (ord("k"), ord("t")):
                baru = kalibrasi(kamera, pelacak, lebar_layar, tinggi_layar, putaran,
                                 tambah=(tombol == ord("t")), uji=not args.tanpa_uji)
                if baru is not None:
                    model = baru
                    ref_buka = model["dasar"]
                saringan = SaringanFiksasi(hitung_radius(model, lebar_layar, tinggi_layar, args.halus))
                klik.batal()
                tertutup, tutup_sejak, terakhir, pra = False, None, None, False


def main():
    bisukan_log_native()
    parser = argparse.ArgumentParser(description="Gerakkan kursor dengan arah pandangan mata")
    parser.add_argument("--kamera", type=int, default=0, help="nomor kamera (bawaan 0)")
    parser.add_argument("--kalibrasi-ulang", action="store_true",
                        help="latih ulang walaupun sudah ada hasil pelatihan tersimpan")
    parser.add_argument("--tambah", action="store_true",
                        help="latih lagi dan tumpuk dengan data sesi sebelumnya (hasil makin akurat)")
    parser.add_argument("--putaran", type=int, default=PUTARAN_BAWAAN,
                        help="jumlah putaran pelatihan (bawaan 3; makin banyak makin akurat tetapi lebih lama)")
    parser.add_argument("--tanpa-uji", action="store_true",
                        help="lewati uji akurasi setelah pelatihan")
    parser.add_argument("--halus", type=float, default=1.0,
                        help="pengali radius fiksasi: kecil = kursor lebih cepat menyusul, "
                             "besar = lebih stabil (bawaan 1.0)")
    parser.add_argument("--respons", type=float, default=None, help=argparse.SUPPRESS)  # kompatibilitas lama
    parser.add_argument("--tanpa-klik", action="store_true", help="matikan klik dengan kedip")
    parser.add_argument("--kedip-min", type=float, default=KEDIP_MIN_BAWAAN,
                        help="durasi kedip terpendek yang dihitung sebagai klik, dalam detik")
    parser.add_argument("--kedip-maks", type=float, default=KEDIP_MAKS_BAWAAN,
                        help="durasi kedip pendek terpanjang, dalam detik")
    parser.add_argument("--kedip-lama", type=float, default=KEDIP_LAMA_BAWAAN,
                        help="durasi mulai dari mana kedip dihitung lama (klik kanan), dalam detik")
    parser.add_argument("--jendela-ganda", type=float, default=JENDELA_GANDA_BAWAAN,
                        help="waktu tunggu kedip kedua untuk klik ganda, dalam detik")
    parser.add_argument("--muat", action="store_true", help=argparse.SUPPRESS)  # kompatibilitas lama
    args = parser.parse_args()
    putaran = max(1, min(args.putaran, 6))

    periksa_izin_mac()
    lebar_layar, tinggi_layar = pyautogui.size()

    kamera = cv2.VideoCapture(args.kamera)
    if not kamera.isOpened():
        raise SystemExit("Kamera tidak bisa dibuka. Periksa izin kamera atau coba --kamera 1.")
    kamera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    kamera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    kamera.set(cv2.CAP_PROP_FPS, 30)

    try:
        jalankan(args, kamera, lebar_layar, tinggi_layar, putaran)
    except KameraMati as galat:
        raise SystemExit(str(galat))
    finally:
        kamera.release()
        cv2.destroyAllWindows()
        for _ in range(4):
            cv2.waitKey(1)


if __name__ == "__main__":
    main()
