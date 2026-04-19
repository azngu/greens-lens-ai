import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

class GreenDigitalAI:
    """
    Hệ thống phân cụm hành vi số với độ nhạy cao.
    Sử dụng MinMaxScaler để phóng đại sự khác biệt của các biến số.
    """
    def __init__(self, n_clusters=5):
        # K=5 giúp tạo ra nhiều ranh giới cụm, khiến 10-20p dễ làm thay đổi kết quả
        self.n_clusters = n_clusters
        self.model = KMeans(n_clusters=self.n_clusters, random_state=42, n_init=20)
        self.scaler = MinMaxScaler()
        self.is_fitted = False

    def _generate_dense_data(self):
        """
        Tạo bộ dữ liệu huấn luyện dày đặc để AI nhận diện được các mốc thời gian sát nhau.
        """
        data = {
            'Social_Time': [0, 15, 30, 45, 60, 80, 100, 120, 150, 180, 210, 240, 300, 400, 600],
            'Study_Time':  [600, 580, 550, 520, 500, 450, 400, 350, 300, 250, 200, 150, 100, 50, 0]
        }
        return pd.DataFrame(data)

    def train_model(self):
        """Huấn luyện mô hình với dữ liệu đã chuẩn hóa"""
        df = self._generate_dense_data()
        # Chuẩn hóa về khoảng [0, 1] để tăng độ nhạy
        scaled_data = self.scaler.fit_transform(df)
        self.model.fit(scaled_data)
        self.is_fitted = True
        
        # Sắp xếp các tâm cụm để dán nhãn theo thứ tự từ 'Tốt' đến 'Xấu'
        # Điều này giúp đảm bảo khi Social_Time tăng, nhãn sẽ chuyển dần sang mức tệ hơn
        self.cluster_order = np.argsort(self.model.cluster_centers_[:, 0]) 
        print("✅ Hệ thống AI đã sẵn sàng với độ nhạy cao.")

    def predict(self, social_time, study_time):
        """Dự đoán nhóm dựa trên dữ liệu thời gian thực"""
        if not self.is_fitted:
            return "Lỗi: Mô hình chưa được huấn luyện."

        # 1. Xử lý đầu vào
        input_data = np.array([[social_time, study_time]])
        input_scaled = self.scaler.transform(input_data)
        
        # 2. Tìm ID cụm thô
        raw_cluster = self.model.predict(input_scaled)[0]
        
        # 3. Chuyển đổi ID cụm sang thứ tự logic (0: Tốt nhất -> 4: Tệ nhất)
        ordered_rank = np.where(self.cluster_order == raw_cluster)[0][0]
        
        # 4. Dán nhãn thông minh
        labels = {
            0: {"name": "Hạt mầm xanh", "color": "Success", "emoji": "🌱"},
            1: {"name": "Lối sống cân bằng", "color": "Primary", "emoji": "⚖️"},
            2: {"name": "Vùng xám (Cần chú ý)", "color": "Warning", "emoji": "⚠️"},
            3: {"name": "Xao nhãng cao", "color": "Orange", "emoji": "📱"},
            4: {"name": "Báo động đỏ (Khói số)", "color": "Danger", "emoji": "🚨"}
        }
        return labels.get(ordered_rank)

# --- CHƯƠNG TRÌNH KIỂM THỬ (DEBUG) ---
if __name__ == "__main__":
    app_ai = GreenDigitalAI()
    app_ai.train_model()
    
    # Kiểm tra độ nhạy: Thay đổi chỉ 20 phút
    test_cases = [(60, 300), (80, 300), (100, 300)]
    for s, st in test_cases:
        res = app_ai.predict(s, st)
        print(f"Social: {s}p, Study: {st}p => Nhóm: {res['name']} {res['emoji']}")
