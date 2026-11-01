import streamlit as st
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np

st.set_page_config(page_title="نظام فرز وتصنيف الرطب", page_icon="🌴", layout="centered")

st.title("🌴 نظام فرز وتصنيف التمر الذكي")

@st.cache_resource
def load_trained_model():
    model = tf.keras.models.load_model('rutab_sorter_model.h5')
    return model

with st.spinner('⏳ جاري تحميل نموذج الذكاء الاصطناعي...'):
    model = load_trained_model()

uploaded_file = st.file_uploader("اختر صورة التمرة...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # تحميل وضبط مقاس الصورة مباشرة (بدون قسمة أو معالجة خارجية)
    image = load_img(uploaded_file, target_size=(224, 224))
    
    st.image(image, caption='الصورة المرفوعة للفحص', use_container_width=True)
    
    # تحويل الصورة إلى مصفوفة (القيم تبقى من 0 إلى 255 كما تدرب عليها النموذج)
    x = img_to_array(image)
    x = np.expand_dims(x, axis=0)
    
    # إجراء التنبؤ
    with st.spinner('🔍 جاري التحليل...'):
        pred = model.predict(x)
        pred_value = float(pred[0][0])
    
    st.markdown("---")
    st.subheader("📊 النتيجة والتحليل:")
    
    st.write(f"🔢 القيمة الخام من النموذج: `{pred_value:.4f}`")
    
    if pred_value > 0.5:
        st.success(f"✅ النتيجة: التمر جيد (Good) - نسبة الثقة: {pred_value * 100:.1f}%")
    else:
        st.error(f"❌ النتيجة: التمر فاسد / سيء (Bad) - نسبة الثقة: {(1 - pred_value) * 100:.1f}%")
