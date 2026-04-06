# app.py

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
    "MKSC-01": "textures/MKSC-01.jpg",
    "MKSC-03": "textures/MKSC-03.jpg",
    "MKSC-05": "textures/MKSC-05.jpg",
    "MKSC-07": "textures/MKSC-07.jpg",
    "MKSC-09": "textures/MKSC-09.jpg",
    "MKSC-10": "textures/MKSC-10.jpg",
    "MKSC-11": "textures/MKSC-11.jpg",
    "MKSC-12": "textures/MKSC-12.jpg",
}

# =============================================
# LOAD MODEL
# =============================================
@st.cache_resource
def load_model():
    return ort.InferenceSession("best.onnx", providers=["CPUExecutionProvider"])

session = load_model()
input_name = session.get_inputs()[0].name

# =============================================
# HELPER FUNCTIONS
# =============================================
def preprocess_image(img_bgr, imgsz=640):
    img_resized = cv2.resize(img_bgr, (imgsz, imgsz))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    img_transposed = np.transpose(img_norm, (2, 0, 1))
    return np.expand_dims(img_transposed, axis=0)

def get_floor_mask(session, img_bgr, conf_threshold=0.25):
    orig_h, orig_w = img_bgr.shape[:2]
    imgsz = 640

    inp = preprocess_image(img_bgr, imgsz)
    outputs = session.run(None, {input_name: inp})

    detections = outputs[0][0]
    proto = outputs[1][0]

    mask_combined = np.zeros((imgsz, imgsz), dtype=np.float32)
    found = False

    for det in detections:
        conf = det[4]
        cls = int(det[5])
        if conf < conf_threshold or cls != 0:
            continue

        found = True
        cx, cy, w, h = det[0], det[1], det[2], det[3]
        mask_coef = det[6:]

        mask = np.einsum('c,chw->hw', mask_coef, proto)
        mask = 1 / (1 + np.exp(-mask))

        x1 = int((cx - w / 2) / imgsz * 160)
        y1 = int((cy - h / 2) / imgsz * 160)
        x2 = int((cx + w / 2) / imgsz * 160)
        y2 = int((cy + h / 2) / imgsz * 160)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(160, x2), min(160, y2)

        mask_crop = np.zeros((160, 160), dtype=np.float32)
        mask_crop[y1:y2, x1:x2] = mask[y1:y2, x1:x2]

        mask_full = cv2.resize(mask_crop, (imgsz, imgsz))
        mask_combined = np.maximum(mask_combined, mask_full)

    if not found:
        return None

    mask_orig = cv2.resize(mask_combined, (orig_w, orig_h))
    binary_mask = (mask_orig > 0.5).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
    binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

    return binary_mask

def order_points(pts):
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def apply_texture_perspective(img_bgr, mask, texture_bgr):
    orig_h, orig_w = img_bgr.shape[:2]

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img_bgr

    largest = max(contours, key=cv2.contourArea)

    epsilon = 0.02 * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon, True)

    if len(approx) == 4:
        dst_pts = order_points(approx.reshape(4, 2).astype(np.float32))
    else:
        hull = cv2.convexHull(largest)
        hull_pts = hull.reshape(-1, 2).astype(np.float32)
        s = hull_pts.sum(axis=1)
        diff = np.diff(hull_pts, axis=1).flatten()
        tl = hull_pts[np.argmin(s)]
        br = hull_pts[np.argmax(s)]
        tr = hull_pts[np.argmin(diff)]
        bl = hull_pts[np.argmax(diff)]
        dst_pts = np.array([tl, tr, br, bl], dtype=np.float32)

    w_top   = np.linalg.norm(dst_pts[1] - dst_pts[0])
    w_bot   = np.linalg.norm(dst_pts[2] - dst_pts[3])
    h_left  = np.linalg.norm(dst_pts[3] - dst_pts[0])
    h_right = np.linalg.norm(dst_pts[2] - dst_pts[1])
    max_w   = max(int(max(w_top, w_bot)), 1)
    max_h   = max(int(max(h_left, h_right)), 1)

    src_pts = np.array([
        [0,         0        ],
        [max_w - 1, 0        ],
        [max_w - 1, max_h - 1],
        [0,         max_h - 1]
    ], dtype=np.float32)

    texture_resized = cv2.resize(texture_bgr, (max_w, max_h))

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    texture_warped = cv2.warpPerspective(texture_resized, M, (orig_w, orig_h))

    mask_blur = cv2.GaussianBlur(mask.astype(np.float32), (21, 21), 0)
    mask_3ch  = np.stack([mask_blur] * 3, axis=-1)
    result    = (mask_3ch * texture_warped + (1 - mask_3ch) * img_bgr).astype(np.uint8)

    return result

# =============================================
# UI
# =============================================
st.title("🏠 Floor Texture Replacer")
st.write("Upload foto ruangan, pilih tekstur lantai, lalu lihat hasilnya.")

room_file = st.file_uploader("📷 Upload foto ruangan", type=["jpg", "jpeg", "png"])

st.subheader("Pilih tekstur lantai")

# Tampilkan preview tekstur dalam grid
cols = st.columns(4)
selected_texture = st.session_state.get("selected_texture", "MKSC-01")

for i, (name, path) in enumerate(TEXTURES.items()):
    with cols[i % 4]:
        if os.path.exists(path):
            st.image(path, caption=name, use_container_width=True)
        if st.button(name, key=f"btn_{name}", use_container_width=True):
            st.session_state["selected_texture"] = name
            selected_texture = name

st.info(f"Tekstur dipilih: **{selected_texture}**")

conf_threshold = st.slider(
    "Sensitivitas deteksi", 0.10, 0.90, 0.25, 0.05,
    help="Turunkan jika lantai tidak terdeteksi, naikkan jika ada objek lain ikut terdeteksi"
)

if room_file:
    room_img = Image.open(room_file).convert("RGB")
    room_bgr = cv2.cvtColor(np.array(room_img), cv2.COLOR_RGB2BGR)

    texture_path = TEXTURES[selected_texture]
    texture_bgr  = cv2.imread(texture_path)

    if texture_bgr is None:
        st.error(f"File tekstur {texture_path} tidak ditemukan.")
    else:
        if st.button("Terapkan Tekstur", type="primary", use_container_width=True):
            with st.spinner("Mendeteksi lantai..."):
                mask = get_floor_mask(session, room_bgr, conf_threshold=conf_threshold)

            if mask is None:
                st.warning("Lantai tidak terdeteksi. Coba turunkan sensitivitas deteksi.")
            else:
                with st.spinner("Menerapkan tekstur..."):
                    result_bgr = apply_texture_perspective(room_bgr, mask, texture_bgr)
                    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
                    result_pil = Image.fromarray(result_rgb)

                st.subheader("Hasil")
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