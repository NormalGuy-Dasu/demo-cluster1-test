"""Test build of the cluster 1 demo: Module 1 (crop, deskew, blank, orientation, barcodes) on a camera photo.

Thin wrapper only: it calls the project code unchanged. One shared model, one visitor at a time,
size limits, and the photo is processed in memory only (nothing is written to disk, D-019).
"""
import gc
import os
import resource
import shutil
import sys
import threading
import time
from pathlib import Path

import streamlit as st

# The project does not use alpha matting; this avoids 30-60 s of numba compile at every start.
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

MAX_UPLOAD_MB = 15
MAX_MEGAPIXELS = 30
MIN_SIDE = 50
SHOW_MAX_SIDE = 1400


def rss_mb():
    import psutil

    return round(psutil.Process().memory_info().rss / 1024 / 1024)


def peak_mb():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)


def seed_models():
    """Put the bundled u2netp model where rembg looks for it, before Module 1 is imported."""
    src = ROOT / "models" / "u2netp.onnx"
    target = Path.home() / ".u2net" / "u2netp.onnx"
    if src.exists() and not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)


@st.cache_resource(show_spinner="Loading Module 1 (first start only)...")
def load_engine():
    t = time.perf_counter()
    seed_models()
    from module_1_image_preprocessing.preprocessor import ImagePreprocessor

    pre = ImagePreprocessor.from_yaml(ROOT / "configs" / "module1_defaults.yaml")
    return {"pre": pre, "lock": threading.Lock(), "load_s": round(time.perf_counter() - t, 1)}


def make_synthetic_photo(kind, tilt_deg, width=3000, height=4000):
    """A drawn photo of a page on a dark table. No personal data."""
    import cv2
    import numpy as np

    img = np.full((height, width, 3), (60, 70, 85), np.uint8)
    if kind == "No paper (plain table)":
        return img
    page = np.full((int(height * 0.8), int(width * 0.8), 3), 245, np.uint8)
    if kind != "Blank page":
        for i in range(30):
            cv2.putText(page, f"Line {i + 1}: Giay chung nhan quyen su dung dat so {1000 + i * 7}", (60, 110 + i * 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.6, (20, 20, 20), 3, cv2.LINE_AA)
    ph, pw = page.shape[:2]
    y0, x0 = (height - ph) // 2, (width - pw) // 2
    img[y0:y0 + ph, x0:x0 + pw] = page
    m = cv2.getRotationMatrix2D((width / 2, height / 2), tilt_deg, 1.0)
    return cv2.warpAffine(img, m, (width, height), borderValue=(60, 70, 85))


def to_jpeg(bgr):
    """Reduced JPEG for the screen, so the page stays light."""
    import cv2

    h, w = bgr.shape[:2]
    s = SHOW_MAX_SIDE / max(h, w)
    if s < 1:
        bgr = cv2.resize(bgr, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes()


def run_module1(data, eng):
    """Image bytes -> Module 1, all in memory. Returns a dict."""
    import cv2
    import numpy as np
    from PIL import Image
    import io

    t_all = time.perf_counter()
    try:
        w, h = Image.open(io.BytesIO(data)).size
    except Exception:  # noqa: BLE001
        return {"error": "bad_image"}
    if w * h > MAX_MEGAPIXELS * 1_000_000:
        return {"error": "too_big", "mp": round(w * h / 1e6, 1)}
    if min(w, h) < MIN_SIDE:
        return {"error": "too_small"}
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "bad_image"}
    t = time.perf_counter()
    res = eng["pre"].process(img)
    t_m1 = time.perf_counter() - t
    out = {
        "before": to_jpeg(img),
        "after": to_jpeg(res.processed_image) if res.processed_image is not None else None,
        "is_blank": res.is_blank,
        "orientation": res.orientation_status.value,
        "warnings": list(res.warnings),
        "barcodes": [{"type": b.barcode_type, "data": b.data} for b in res.barcodes],
        "error_code": res.error_code,
        "stats": {
            "size_px": f"{w}x{h}", "megapixels": round(w * h / 1e6, 1), "skew_angle": round(float(res.skew_angle), 2),
            "module1_s": round(t_m1, 2), "total_s": round(time.perf_counter() - t_all, 2),
            "rss_now_mb": rss_mb(), "peak_rss_mb": peak_mb(), "files_written": 0,
        },
    }
    del img, res
    gc.collect()
    return out


def guarded_run(data, eng):
    """One visitor at a time; the lock is always released."""
    if not eng["lock"].acquire(blocking=False):
        st.warning("Đang xử lý cho một người dùng khác. Vui lòng thử lại sau ít phút.")
        return None
    try:
        with st.spinner("Processing..."):
            return run_module1(data, eng)
    finally:
        eng["lock"].release()


def show_result(r):
    err = r.get("error")
    if err == "bad_image":
        st.error("Không đọc được file này. Hãy chọn một ảnh JPG hoặc PNG hợp lệ.")
        return
    if err == "too_big":
        st.error(f"Ảnh {r['mp']} MP, vượt giới hạn {MAX_MEGAPIXELS} MP. Hãy chụp ở độ phân giải thấp hơn.")
        return
    if err == "too_small":
        st.error(f"Ảnh quá nhỏ (cạnh ngắn dưới {MIN_SIDE} px).")
        return
    if r.get("error_code"):
        st.error(f"Module 1 báo lỗi: {r['error_code']}")
    if "CROP_FAILED_PLEASE_RETAKE" in r["warnings"]:
        st.warning("Không tìm thấy rõ tờ giấy trong ảnh nên giữ nguyên ảnh gốc. Hãy chụp lại: tờ giấy nằm trọn trong khung, nền tương phản.")
    if r["is_blank"]:
        st.warning("Trang này có vẻ trống.")
    if r["orientation"] != "LIKELY_CORRECT":
        st.info(f"Hướng chữ: {r['orientation']} (chưa chắc là đúng chiều).")
    if r["barcodes"]:
        st.success(f"Đọc được {len(r['barcodes'])} mã vạch/QR.")
        st.table(r["barcodes"])
    c1, c2 = st.columns(2)
    c1.caption("Before")
    c1.image(r["before"])
    c2.caption("After (cropped and straightened)")
    if r["after"]:
        c2.image(r["after"])
    with st.expander("Diagnostics"):
        st.json(r["stats"])


st.set_page_config(page_title="Camera-photo preprocessing test", layout="wide")
st.title("Camera-photo preprocessing (test build)")
st.caption("Module 1 on a photo of a document. Test build: do not upload photos with personal data. The photo is processed in memory and nothing is stored.")

engine = load_engine()
st.caption(f"Module 1 loaded in {engine['load_s']} s. Memory now {rss_mb()} MB, peak {peak_mb()} MB (limit about 3000 MB).")

if "last" not in st.session_state:
    st.session_state["last"] = None
    st.session_state["history"] = []

tab_up, tab_syn = st.tabs(["Upload a photo", "Synthetic test photo"])

with tab_up:
    up = st.file_uploader("Photo of a document (JPG or PNG)", type=["jpg", "jpeg", "png"])
    if up is not None and st.button("Run Module 1 on the uploaded photo"):
        if up.size > MAX_UPLOAD_MB * 1024 * 1024:
            st.session_state["last"] = {"error": "too_big_file"}
            st.error(f"File lớn hơn {MAX_UPLOAD_MB} MB.")
        else:
            res = guarded_run(up.getvalue(), engine)
            if res is not None:
                st.session_state["last"] = res
                st.session_state["history"] = []

with tab_syn:
    st.write("Draws a 12-megapixel photo with no personal data and runs the same path.")
    c1, c2, c3 = st.columns(3)
    kind = c1.radio("Photo", ["Tilted page", "Blank page", "No paper (plain table)"])
    tilt = c2.slider("Tilt (degrees)", -25, 25, 12)
    repeats = c3.number_input("Repeat runs", 1, 10, 1)
    if st.button("Run synthetic test"):
        import cv2

        ok, buf = cv2.imencode(".jpg", make_synthetic_photo(kind, tilt), [cv2.IMWRITE_JPEG_QUALITY, 90])
        data = buf.tobytes()
        history, last = [], None
        for k in range(int(repeats)):
            last = guarded_run(data, engine)
            if last is None:
                break
            if "stats" in last:
                history.append({"run": k + 1, **{x: last["stats"][x] for x in ("megapixels", "skew_angle", "module1_s", "total_s", "rss_now_mb", "peak_rss_mb")}})
        if last is not None:
            st.session_state["last"] = last
            st.session_state["history"] = history

if st.session_state["history"]:
    st.table(st.session_state["history"])
if st.session_state["last"] is not None and st.session_state["last"].get("error") != "too_big_file":
    show_result(st.session_state["last"])
