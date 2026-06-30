import streamlit as st
from PIL import Image
from io import BytesIO
import zipfile
import numpy as np
import cv2

st.set_page_config(page_title="Shaper Tools - Fotos Pro", layout="wide")

st.title("Shaper Tools - Fotos Pro")
st.write("Mejora automática + borrar marca anterior + marca Shaper centrada + máxima calidad.")

photos = st.file_uploader(
    "1. Sube tus fotos",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

watermark_file = st.file_uploader(
    "2. Sube tu marca de agua Shaper PNG",
    type=["png"]
)

st.subheader("Ajustes Pro")

auto_enhance = st.checkbox("Embellecer automáticamente", value=True)

remove_old_watermark = st.checkbox("Borrar marca de agua anterior", value=False)

old_wm_position = st.selectbox(
    "Ubicación marca anterior",
    ["Centro", "Abajo derecha", "Abajo izquierda", "Arriba derecha", "Arriba izquierda"],
    index=0
)

old_wm_width = st.slider("Ancho zona a borrar (% de la foto)", 10, 90, 45)
old_wm_height = st.slider("Alto zona a borrar (% de la foto)", 5, 60, 18)

brightness = st.slider("Luz extra", 0.90, 1.30, 1.08, 0.01)
contrast_strength = st.slider("Contraste local", 0.5, 3.0, 1.4, 0.1)
saturation = st.slider("Color / Saturación", 0.80, 1.40, 1.08, 0.01)
sharpness = st.slider("Nitidez inteligente", 0.0, 2.0, 0.65, 0.05)
denoise = st.slider("Reducción de ruido", 0, 20, 5)

st.subheader("Marca de agua Shaper")

opacity = st.slider("Opacidad marca Shaper", 0.05, 1.00, 0.35, 0.01)
scale = st.slider("Tamaño marca Shaper (% del ancho)", 5, 90, 65)

st.subheader("Exportación")

jpeg_quality = st.slider("Calidad JPG", 90, 100, 100)


def pil_to_cv(image):
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def cv_to_pil(image):
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def automatic_real_estate_enhance(pil_image):
    image = pil_to_cv(pil_image)

    if denoise > 0:
        image = cv2.fastNlMeansDenoisingColored(
            image,
            None,
            denoise,
            denoise,
            7,
            21
        )

    result = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])

    result[:, :, 1] = result[:, :, 1] - (
        (avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1
    )
    result[:, :, 2] = result[:, :, 2] - (
        (avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1
    )

    image = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=contrast_strength,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    image = cv2.convertScaleAbs(image, alpha=brightness, beta=4)

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = hsv[:, :, 1] * saturation
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    hsv = hsv.astype(np.uint8)

    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    if sharpness > 0:
        blurred = cv2.GaussianBlur(image, (0, 0), 1.2)
        image = cv2.addWeighted(image, 1 + sharpness, blurred, -sharpness, 0)

    return cv_to_pil(image)


def remove_watermark_area(pil_image):
    image = pil_to_cv(pil_image)
    h, w = image.shape[:2]

    box_w = int(w * old_wm_width / 100)
    box_h = int(h * old_wm_height / 100)
    margin = int(w * 0.03)

    if old_wm_position == "Centro":
        x1 = (w - box_w) // 2
        y1 = (h - box_h) // 2
    elif old_wm_position == "Abajo derecha":
        x1 = w - box_w - margin
        y1 = h - box_h - margin
    elif old_wm_position == "Abajo izquierda":
        x1 = margin
        y1 = h - box_h - margin
    elif old_wm_position == "Arriba derecha":
        x1 = w - box_w - margin
        y1 = margin
    else:
        x1 = margin
        y1 = margin

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(x1 + box_w, w)
    y2 = min(y1 + box_h, h)

    mask = np.zeros((h, w), dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255

    result = cv2.inpaint(image, mask, 7, cv2.INPAINT_TELEA)

    return cv_to_pil(result)


def apply_watermark(pil_image, watermark_file):
    image = pil_image.convert("RGBA")
    watermark = Image.open(watermark_file).convert("RGBA")

    img_w, img_h = image.size

    wm_width = int(img_w * scale / 100)
    wm_ratio = wm_width / watermark.width
    wm_height = int(watermark.height * wm_ratio)

    watermark = watermark.resize((wm_width, wm_height), Image.LANCZOS)

    alpha = watermark.split()[3]
    alpha = alpha.point(lambda p: int(p * opacity))
    watermark.putalpha(alpha)

    x = (img_w - wm_width) // 2
    y = (img_h - wm_height) // 2

    image.paste(watermark, (x, y), watermark)

    return image.convert("RGB")


def process_photo(photo, watermark_file):
    image = Image.open(photo).convert("RGB")

    if remove_old_watermark:
        image = remove_watermark_area(image)

    if auto_enhance:
        image = automatic_real_estate_enhance(image)

    image = apply_watermark(image, watermark_file)

    return image


if photos and watermark_file:
    st.success("Listo. Sube una foto de prueba y ajusta la zona si vas a borrar una marca anterior.")

    col1, col2 = st.columns(2)

    original_preview = Image.open(photos[0]).convert("RGB")
    processed_preview = process_photo(photos[0], watermark_file)

    with col1:
        st.subheader("Antes")
        st.image(original_preview, use_container_width=True)

    with col2:
        st.subheader("Después")
        st.image(processed_preview, use_container_width=True)

    if st.button("Procesar todas las fotos"):
        zip_buffer = BytesIO()

        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for photo in photos:
                result = process_photo(photo, watermark_file)

                output = BytesIO()
                result.save(
                    output,
                    format="JPEG",
                    quality=jpeg_quality,
                    subsampling=0
                )

                filename = photo.name.rsplit(".", 1)[0] + "_shaper_pro.jpg"
                zip_file.writestr(filename, output.getvalue())

        st.download_button(
            label="Descargar ZIP",
            data=zip_buffer.getvalue(),
            file_name="fotos_shaper_pro.zip",
            mime="application/zip"
        )
else:
    st.warning("Sube fotos y una marca de agua PNG para comenzar.")
