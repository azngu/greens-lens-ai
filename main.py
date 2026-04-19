import pandas as pd
from sklearn.cluster import KMeans
import numpy as np

class GreenDigitalAI:
    def __init__(self):
        """
        Khởi tạo Engine AI cho dự án Green Lens.
        Dữ liệu huấn luyện được mô phỏng dựa trên 30 mẫu khảo sát thực tế (MTC 2.c).
        """
        # 1. TẠO DATASET HUẤN LUYỆN (Khớp với báo cáo số 2)
        self.raw_data = {
            'Social_Time': [45, 300, 120, 500, 60, 210, 30, 400, 150, 80, 320, 45, 180, 550, 100, 
                            250, 15, 420, 130, 90, 310, 50, 200, 480, 110, 280, 20, 450, 140, 350],
            'Study_Time':  [300, 60, 180, 30, 240, 150, 350, 20, 200, 120, 50, 400, 150, 10, 280,
                            100, 420, 40, 210, 160, 70, 330, 140, 45, 250, 90, 380, 55, 190, 30]
        }
        self.df = pd.DataFrame(self.raw_data)
        
        # 2. KHỞI TẠO MÔ HÌNH K-MEANS (Chia làm 3 cụm)
        self.model = KMeans(n_clusters=3, random_state=42, n_init=10)
        self.train_model()

    def train_model(self):
        """Huấn luyện mô hình từ dữ liệu sạch"""
        self.model.fit(self.df)

    def calculate_carbon(self, social_time):
        """
        Tính toán lượng Carbon phát thải (MTC 3.e)
        Công thức: 1 phút online ≈ 0.15g CO2
        """
        return round(social_time * 0.15, 2)

    def predict_group(self, social_time, study_time):
        """
        Dự đoán nhóm hành vi dựa trên khoảng cách đến các tâm cụm (Centroids)
        """
        # Chuẩn bị dữ liệu đầu vào
        user_input = np.array([[social_time, study_time]])
        
        # Dự đoán cụm (0, 1, hoặc 2)
        cluster_id = self.model.predict(user_input)[0]
        
        # Lấy tọa độ tâm cụm để xác định tính chất (Phản biện logic MTC 2.e)
        centroids = self.model.cluster_centers_
        avg_social = centroids[cluster_id][0]
        avg_study = centroids[cluster_id][1]

        # Gán nhãn dựa trên đặc điểm của tâm cụm
        if avg_social < 150 and avg_study > 250:
            return "🌱 Hạt mầm xanh", "Tích cực & Bền vững", "green"
        elif avg_social > 350:
            return "🚨 Báo động đỏ", "Xao nhãng & Nguy cơ nghiện số", "red"
        else:
            return "⚠️ Vùng xám", "Cân bằng hoặc Cần cải thiện", "orange"

# Khởi tạo đối tượng duy nhất để app.py sử dụng (Singleton Pattern)
ai_engine = GreenDigitalAI()
