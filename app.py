import streamlit as st
import onnxruntime as ort
import numpy as np
import cv2
from PIL import Image
import io
import os

st.set_page_config(
    page_title="Floor Texture Replacer",
    page_icon="🏠",
    layout="centered"
)

TEXTURES = {
    "MKSC-01": "textures/MKSC-01.png",
    "MKSC-03": "textures/MKSC-03.png",
    "MKSC-05": "textures/MKSC-05.png",
    "MKSC-07": "textures/MKSC-07.png",
    "MKSC-09": "textures/MKSC-09.png",
    "MKSC-10": "textures/MKSC-10.png",
    "MKSC-11": "textures/MKSC-11.png",
    "MKSC-12": "textures/MKSC-12.png",
}

MAX_FILE_SIZE_MB = 10
MAX_DIMENSION    = 4096
ALLOWED_TYPES    = {"jpg", "jpeg", "png"}

@st.cache_resource
def load_model():
    if not os.path.exists("best.onnx"):
        st.error("Model best.onnx tidak ditemukan.")
        st.stop()
    return ort.InferenceSession("best.onnx", providers=["CPUExecutionProvider"])

session   = load_model()
input_name = session.get_inputs()[0].name

def validate_image(uploaded_file):
    if uploaded_file is None:
        return None, "File tidak ditemukan."

    ext = uploaded_file.name.rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED_TYPES:
        return None, f"Format tidak didukung: .{ext}. Gunakan JPG atau PNG."

    size_mb = uploaded_file.size / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        return None, f"Ukuran file terlalu besar ({size_mb:.1f} MB). Maksimal {MAX_FILE_SIZE_MB} MB."

    try:
        img = Image.open(uploaded_file).convert("RGB")
        w, h = img.size
        if w > MAX_DIMENSION or h > MAX_DIMENSION:
            return None, f"Resolusi terlalu besar ({w}x{h}). Maksimal {MAX_DIMENSION}px."
        if w < 64 or h < 64:
            return None, f"Gambar terlalu kecil ({w}x{h}). Minimal 64x64px."
        return img, None
    except Exception:
        return None, "File rusak atau bukan gambar yang valid."

def validate_texture(name):
    if name not in TEXTURES:
        return None, "Tekstur tidak valid."
    path = TEXTURES[name]
    if not os.path.exists(path):
        return None, f"File tekstur {name} tidak ditemukan di server."
    texture = cv2.imread(path)
    if texture is None:
        return None, f"File tekstur {name} tidak bisa dibaca."
    return texture, None

def preprocess_image(img_bgr, imgsz=640):
    img_resized    = cv2.resize(img_bgr, (imgsz, imgsz))
    img_rgb        = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_norm       = img_rgb.astype(np.float32) / 255.0
    img_transposed = np.transpose(img_norm, (2, 0, 1))
    return np.expand_dims(img_transposed, axis=0)

def get_floor_mask(session, img_bgr, conf_threshold=0.25):
    orig_h, orig_w = img_bgr.shape[:2]
    imgsz = 640

    inp     = preprocess_image(img_bgr, imgsz)
    outputs = session.run(None, {input_name: inp})

    detections = outputs[0][0].transpose(1, 0)
    proto      = outputs[1][0]  # (32, 160, 160)

    mask_combined = np.zeros((imgsz, imgsz), dtype=np.float32)
    found = False

    for det in detections:
        cx, cy, w, h = float(det[0]), float(det[1]), float(det[2]), float(det[3])
        cls_score    = float(det[4])

        if cls_score < conf_threshold:
            continue

        mask_coef = det[5:37]
        if mask_coef.shape[0] != 32:
            continue

        found = True

        mask_raw = np.einsum('c,chw->hw', mask_coef, proto)
        mask_sig = 1 / (1 + np.exp(-mask_raw))

        x1 = max(0,   int((cx - w / 2) / imgsz * 160))
        y1 = max(0,   int((cy - h / 2) / imgsz * 160))
        x2 = min(160, int((cx + w / 2) / imgsz * 160))
        y2 = min(160, int((cy + h / 2) / imgsz * 160))

        if x2 <= x1 or y2 <= y1:
            continue

        mask_crop          = np.zeros((160, 160), dtype=np.float32)
        mask_crop[y1:y2, x1:x2] = mask_sig[y1:y2, x1:x2]
        mask_full          = cv2.resize(mask_crop, (imgsz, imgsz))
        mask_combined      = np.maximum(mask_combined, mask_full)

    if not found:
        return None

    mask_orig   = cv2.resize(mask_combined, (orig_w, orig_h))
    binary_mask = (mask_orig > 0.5).astype(np.uint8)

    kernel      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

    return binary_mask

def order_points(pts):
    rect      = np.zeros((4, 2), dtype=np.float32)
    s         = pts.sum(axis=1)
    diff      = np.diff(pts, axis=1)
    rect[0]   = pts[np.argmin(s)]
    rect[2]   = pts[np.argmax(s)]
    rect[1]   = pts[np.argmin(diff)]
    rect[3]   = pts[np.argmax(diff)]
    return rect

def apply_texture_perspective(img_bgr, mask, texture_bgr):
    orig_h, orig_w = img_bgr.shape[:2]

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img_bgr

    largest = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(largest, True)
    approx  = cv2.approxPolyDP(largest, epsilon, True)

    if len(approx) == 4:
        dst_pts = order_points(approx.reshape(4, 2).astype(np.float32))
    else:
        hull     = cv2.convexHull(largest)
        hull_pts = hull.reshape(-1, 2).astype(np.float32)
        s        = hull_pts.sum(axis=1)
        diff     = np.diff(hull_pts, axis=1).flatten()
        dst_pts  = np.array([
            hull_pts[np.argmin(s)],
            hull_pts[np.argmin(diff)],
            hull_pts[np.argmax(s)],
            hull_pts[np.argmax(diff)],
        ], dtype=np.float32)

    max_w = max(int(max(
        np.linalg.norm(dst_pts[1] - dst_pts[0]),
        np.linalg.norm(dst_pts[2] - dst_pts[3])
    )), 1)
    max_h = max(int(max(
        np.linalg.norm(dst_pts[3] - dst_pts[0]),
        np.linalg.norm(dst_pts[2] - dst_pts[1])
    )), 1)

    src_pts = np.array([
        [0,         0        ],
        [max_w - 1, 0        ],
        [max_w - 1, max_h - 1],
        [0,         max_h - 1],
    ], dtype=np.float32)

    texture_resized = cv2.resize(texture_bgr, (max_w, max_h))
    M               = cv2.getPerspectiveTransform(src_pts, dst_pts)
    texture_warped  = cv2.warpPerspective(texture_resized, M, (orig_w, orig_h))

    mask_blur = cv2.GaussianBlur(mask.astype(np.float32), (21, 21), 0)
    mask_3ch  = np.stack([mask_blur] * 3, axis=-1)
    result    = (mask_3ch * texture_warped + (1 - mask_3ch) * img_bgr).astype(np.uint8)

    return result

st.title("🏠 Floor Texture Replacer")
st.write("Upload foto ruangan, pilih tekstur lantai, lalu lihat hasilnya.")

room_file = st.file_uploader(
    "📷 Upload foto ruangan (JPG/PNG, maks 10 MB)",
    type=list(ALLOWED_TYPES)
)

st.subheader("Pilih tekstur lantai")
cols             = st.columns(4)
selected_texture = st.session_state.get("selected_texture", "MKSC-01")

for i, (name, path) in enumerate(TEXTURES.items()):
    with cols[i % 4]:
        if os.path.exists(path):
            st.image(path, caption=name, width="stretch")
        if st.button(name, key=f"btn_{name}", use_container_width=True):
            st.session_state["selected_texture"] = name
            selected_texture = name

st.info(f"Tekstur dipilih: **{selected_texture}**")

conf_threshold = st.slider(
    "Sensitivitas deteksi", 0.10, 0.90, 0.25, 0.05,
    help="Turunkan jika lantai tidak terdeteksi, naikkan jika ada objek lain ikut terdeteksi"
)

if room_file:
    room_img, err = validate_image(room_file)
    if err:
        st.error(err)
        st.stop()

    texture_bgr, err = validate_texture(selected_texture)
    if err:
        st.error(err)
        st.stop()

    room_bgr = cv2.cvtColor(np.array(room_img), cv2.COLOR_RGB2BGR)

    if st.button("Terapkan Tekstur", type="primary", use_container_width=True):
        with st.spinner("Mendeteksi lantai..."):
            mask = get_floor_mask(session, room_bgr, conf_threshold=conf_threshold)

        if mask is None or mask.sum() == 0:
            st.warning("Lantai tidak terdeteksi. Coba turunkan sensitivitas deteksi atau gunakan foto dengan lantai yang lebih terlihat.")
        else:
            with st.spinner("Menerapkan tekstur..."):
                result_bgr = apply_texture_perspective(room_bgr, mask, texture_bgr)
                result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
                result_pil = Image.fromarray(result_rgb)

            col1, col2 = st.columns(2)
            with col1:
                st.image(room_img, caption="Original", use_container_width=True)
            with col2:
                st.image(result_pil, caption=f"Tekstur {selected_texture}", use_container_width=True)

            buf = io.BytesIO()
            result_pil.save(buf, format="JPEG", quality=95)
            st.download_button(
                label="⬇️ Download hasil",
                data=buf.getvalue(),
                file_name=f"floor_{selected_texture}.jpg",
                mime="image/jpeg"
            )