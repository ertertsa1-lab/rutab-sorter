import streamlit as st
import tensorflow as tf
from tensorflow.keras.utils import img_to_array
import numpy as np
from PIL import Image, ImageOps
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input  # 🔑 الدالة السرية لـ MobileNetV2

st.set_page_config(page_title="نظام فرز وتصنيف الرطب", page_icon="🌴", layout="centered")

st.title("🌴 نظام فرز وتصنيف الرطب الذكي")

@st.cache_resource
def load_trained_model():
    model = tf.keras.models.load_model('rutab_sorter_model.h5')
    return model

with st.spinner('⏳ جاري تحميل نموذج الذكاء الاصطناعي...'):
    model = load_trained_model()

uploaded_file = st.file_uploader("اختر صورة التمرة...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    try:
        image = ImageOps.exif_transpose(image)
    except Exception:
        pass
        
    st.image(image, caption='الصورة المرفوعة للفحص', use_container_width=True)
    
    # 1. ضبط المقاس بدقة
    img_resized = image.resize((224, 224))
    x = img_to_array(img_resized)
    x = np.expand_dims(x, axis=0)
    
    # 🔑 2. المعالجة الرسمية الخاصة بـ MobileNetV2 (بديلة للقسمة على 255)
    x = preprocess_input(x)
    
    # 3. التنبؤ الفعلي
    with st.spinner('🔍 جاري التحليل...'):
        pred = model.predict(x)
        pred_value = float(pred[0][0])
    
    st.markdown("---")
    st.subheader("📊 النتيجة والتحليل:")
    
    st.write(f"🔢 القيمة الرقمية الخام (Raw Score): `{pred_value:.4f}`")
    
    # عكس الشرط لتصحيح الترتيب الأبجدي للفئات
    if pred_value < 0.5:
        st.success(f"✅ **النتيجة: رطب جيد (Good)** (الثقة: {(1 - pred_value) * 100:.1f}%)")
    else:
        st.error(f"❌ **النتيجة: رطب فاسد / سيء (Bad)** (الثقة: {pred_value * 100:.1f}%)")
