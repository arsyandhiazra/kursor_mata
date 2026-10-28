#!/bin/bash
# Klik ganda berkas ini untuk membuka Kursor Mata.
# Letakkan berkas ini satu folder dengan kursor_mata.py dan face_landmarker.task.
cd "$(dirname "$0")" || exit 1

# Pilih Python pertama yang paketnya lengkap (menghindari salah pasang pip/python).
PY=""
for kandidat in "$PWD/.venv/bin/python3" /opt/homebrew/bin/python3 /usr/local/bin/python3 "$(command -v python3)"; do
    if [ -n "$kandidat" ] && [ -x "$kandidat" ] \
        && "$kandidat" -c "import cv2, mediapipe, pyautogui, numpy" >/dev/null 2>&1; then
        PY="$kandidat"
        break
    fi
done

if [ -z "$PY" ]; then
    PY="$(command -v python3)"
    echo "Paket Python belum lengkap (butuh opencv-python, mediapipe, pyautogui, numpy)."
    if [ -n "$PY" ]; then
        echo "Pasang dengan:"
        echo "  $PY -m pip install -U opencv-python mediapipe pyautogui numpy"
    else
        echo "python3 tidak ditemukan. Pasang Python 3 dari python.org atau Homebrew."
    fi
    echo
    echo "Tutup jendela ini dengan Command + W."
    exit 1
fi

"$PY" kursor_mata.py "$@"
echo
echo "Kursor Mata berhenti. Tutup jendela ini dengan Command + W."
