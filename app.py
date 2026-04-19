import streamlit as st
from main import GreenDigitalAI # Nhập Class từ file main.py

# Cấu hình trang
st.set_page_config(page_title="Green Lens AI", page_icon="🌱")

# Khởi tạo AI (Lưu vào session_state để không phải train lại mỗi lần click)
if 'ai' not in st.session_state:
    st.session_state.ai = GreenDigitalAI()
    st.session_state.ai.train_model()

# --- GIAO DIỆN ---
st.title("🌱 Green Lens: Phân Loại Hành Vi Số")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.header("📊 Nhập số liệu")
    social = st.slider("Thời gian mạng xã hội (phút/ngày)", 0, 600, 60, step=5)
    study = st.slider("Thời gian học tập (phút/ngày)", 0, 600, 300, step=5)

with col2:
    st.header("🤖 Kết quả AI")
    if st.button("Phân tích ngay", use_container_width=True):
        result = st.session_state.ai.predict(social, study)
        
        # Hiển thị kết quả trực quan
        st.info(f"Nhóm phân loại: **{result['name']}** {result['emoji']}")
        
        # Thanh tiến trình giả lập mức độ carbon
        progress_val = (social / 600)
        st.write("Mức độ dấu chân Carbon điện tử:")
        st.progress(progress_val)
        
        if social > 120:
            st.warning("Lời khuyên: Bạn đang thải ra lượng CO2 số khá lớn. Hãy 'Detox' 15 phút!")
        else:
            st.success("Tuyệt vời! Bạn đang duy trì lối sống số rất xanh.")

st.markdown("---")
st.caption("Dự án AI lớp 10 - Nhánh 4: Quản lý thời gian & Cuộc sống xanh.")