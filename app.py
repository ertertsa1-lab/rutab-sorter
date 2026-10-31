import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# إعدادات الصفحة لتكون باللغة العربية واتجاه RTL
st.markdown(
    """
    <style>
    body, .stApp {
        direction: rtl;
        text-align: right;
    }
    .css-1544g2n {direction: rtl;}
    </style>
    """,
    unsafe_allow_html=True
)

# 1. تحميل النموذج (تأكد أن ملف rutab_sorter_model.h5 موجود في نفس المجلد)
@st.cache_resource
def load_model():
    import os

# تحديد المسار المطلق لملف النموذج بناءً على مكان ملف app.py
model_path = os.path.join(os.path.dirname(__file__), "rutab_sorter_model.h5")
model = tf.keras.models.load_model(model_path)

try:
    model = load_model()
except:
    st.error("⚠️ تنبيه: لم يتم العثور على ملف النموذج 'rutab_sorter_model.h5'. تأكد من وضعه مع الملفات.")

# رأس الصفحة واختيار اللغة المصغر
col1, col2 = st.columns([4, 1])
with col1:
    st.title("🌴 نظام فحص وفرز الرطب الآلي")
with col2:
    lang = st.selectbox("اللغة / Lang", ["العربية", "English"], label_visibility="collapsed")

if lang == "العربية":
    st.write("ارفع صورة حبة الرطب أو التقطها، وسيخبرك الذكاء الاصطناعي بنتيجتها فوراً.")
    upload_label = "إدخال الصورة (رفع أو كاميرا)"
    btn_label = "فحص الصورة 🔍"
else:
    st.write("Upload or capture a date image, and the AI will predict its quality immediately.")
    upload_label = "Upload Image"
    btn_label = "Predict 🔍"

# تخطيط الواجهة (الصورة على اليمين والنتيجة على اليسار)
col_img, col_res = st.columns(2)

with col_img:
    uploaded_file = st.file_uploader(upload_label, type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="الصورة المدخلة / Uploaded Image", use_column_width=True)

with col_res:
    st.subheader("نتيجة الفحص / Result")
    if st.button(btn_label, type="primary"):
        if uploaded_file is None:
            st.warning("الرجاء رفع صورة أولاً!" if lang == "العربية" else "Please upload an image first!")
        else:
            # معالجة الصورة وفحصها
            img = image.resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            prediction = model.predict(img_array)
            score = prediction[0][0]
            
            # عتبة الصرامة العالية
            strictness_threshold = 0.75 
            is_bad = score < strictness_threshold
            
            if lang == "العربية":
                if is_bad:
                    st.error(f"🔴 النتيجة: هذا الرطب فاسد - يجب استبعاده (النسبة: {score:.2f})")
                else:
                    st.success(f"🟢 النتيجة: هذا الرطب جيد ومناسب للبيع (النسبة: {score:.2f})")
            else:
                if is_bad:
                    st.error(f"🔴 Result: Bad - Needs to be discarded (Score: {score:.2f})")
                else:
                    st.success(f"🟢 Result: Good - Suitable for sale (Score: {score:.2f})")
