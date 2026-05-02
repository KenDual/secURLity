# Tên dự án  
**Ứng dụng mô hình học sâu lai CNN-LSTM kết hợp XGBoost trong việc nhận diện URL độc hại**

---

## 1. Tổng quan Nghiên cứu

### Khung Công nghệ Đề xuất

- **Mô hình lai CNN-LSTM (Phân tích chuỗi ký tự)**:  
  Phương pháp này vận dụng Mạng nơ-ron tích chập (1D-CNN) nhằm trích xuất các đặc trưng cục bộ (local features) từ cấp độ ký tự của chuỗi URL. Tiếp nối, Mạng bộ nhớ dài-ngắn (LSTM) phân tích chuỗi đặc trưng vừa thu được để nắm bắt các phụ thuộc ngữ cảnh dài hạn bên trong cấu trúc URL. Đây là mô hình chính của hệ thống, học trực tiếp từ **biểu diễn thô cấp ký tự** mà không cần bước feature engineering thủ công.

- **XGBoost với Lexical Features (Phân tích thống kê cấu trúc)**:  
  Mô hình thứ hai tiếp cận bài toán theo hướng hoàn toàn khác biệt: thay vì học từ chuỗi ký tự thô, XGBoost học từ một **vector đặc trưng số học được thiết kế có chủ đích** (engineered feature vector), bao gồm các thuộc tính cấu trúc (độ dài URL, độ sâu path, số tham số query), thuộc tính thống kê (mật độ chữ số, entropy Shannon, độ dài hostname), và các cờ ngữ nghĩa (từ khóa đáng ngờ, TLD bất thường, host dạng IP, extension độc hại). Sự đối lập về paradigm giữa hai mô hình — *sequence learning* và *feature-based learning* — là nền tảng học thuật của hệ thống đa mô hình này.

- **AI có thể giải thích (XAI) — SHAP**:  
  Kỹ thuật SHAP (SHapley Additive exPlanations) được tích hợp cho cả hai mô hình. Đối với CNN-LSTM, DeepExplainer/GradientExplainer tạo ra biểu đồ quy kết ký tự. Đối với XGBoost, TreeExplainer (nhanh hơn đáng kể, không cần GPU) tạo ra biểu đồ tầm quan trọng đặc trưng và beeswarm summary plot. Kết hợp hai góc nhìn giải thích này giúp minh bạch hóa toàn diện cơ chế ra quyết định của hệ thống.

- **Hệ thống Đánh giá Rủi ro**:  
  Nền tảng được xây dựng trên Python (FastAPI hoặc Flask), nhận hai score độc lập từ CNN-LSTM và XGBoost, sau đó phân loại rủi ro theo thang bốn cấp độ tiêu chuẩn.

---

## 2. Mục tiêu Nghiên cứu

- Nghiên cứu và xây dựng hệ thống đa mô hình kết hợp CNN-LSTM (sequence learning) và XGBoost (feature-based learning) để phân loại URL độc hại với độ chính xác và tính ổn định cao.
- Áp dụng thành công cơ chế XAI (SHAP) để minh bạch hóa quy trình ra quyết định của cả hai mô hình, cung cấp cơ sở luận lý hỗ trợ phân tích rủi ro an toàn thông tin.
- Phát triển công cụ phần mềm cục bộ hỗ trợ tự động hóa quy trình trích xuất đặc trưng, suy luận mô hình, và xuất báo cáo rủi ro cho một URL mục tiêu.

---

## 3. Đối tượng và Phạm vi Nghiên cứu

- **Đối tượng Nghiên cứu**:  
  Hệ thống phân tích chuỗi ký tự URL ở hai tầng: tầng biểu diễn thô (raw character sequence cho CNN-LSTM) và tầng đặc trưng số học (engineered lexical/statistical features cho XGBoost). Dữ liệu huấn luyện là tập tổng hợp 10 triệu URL (urls_synthetic_10m.csv) với tỉ lệ 85% benign / 15% malicious.

- **Phạm vi Nghiên cứu**:  
  Giới hạn phân loại các hình thức tấn công Web phổ biến: Phishing, Malware, DGA, IP-based C2, Spam/Scam. Kiểm thử và đánh giá hoàn toàn trong môi trường cục bộ (Local/Lab), chưa bao gồm tích hợp Production.

---

## 4. Phương pháp và Quy trình Triển khai

1. **Tiền xử lý Dữ liệu**:  
   Chuẩn hóa URL (lowercase, strip whitespace), tạo stratified 80/10/10 split. Với CNN-LSTM: mã hóa ký tự thành vector số nguyên, pad/truncate về 200 ký tự. Với XGBoost: trích xuất vector đặc trưng gồm ~30 thuộc tính số học và nhị phân từ cấu trúc URL, áp dụng thống kê chỉ từ tập train để tránh data leakage.

2. **Huấn luyện CNN-LSTM**:  
   Các lớp 1D-CNN rà soát n-gram ký tự bất thường; đầu ra tiếp tục được xử lý bởi LSTM để mô hình hóa cấu trúc phân cấp tuần tự. Huấn luyện với early stopping trên validation F1, log metrics qua TensorBoard.

3. **Huấn luyện XGBoost**:  
   Căn chỉnh `scale_pos_weight` theo tỉ lệ mất cân bằng lớp (85:15). Tuning các siêu tham số: `max_depth`, `learning_rate`, `n_estimators`, `subsample`, `colsample_bytree`. Early stopping trên validation F1. Không cần GPU — chạy hoàn toàn trên CPU, không xung đột tài nguyên với CNN-LSTM.

4. **Cơ chế Phân loại Ngưỡng Rủi ro**:  
   Hai mô hình xuất hai xác suất độc lập. Mỗi score được ánh xạ sang bốn nhóm:
   - **Low Risk (0–25%)**: Mức độ an toàn cao  
   - **Caution (26–50%)**: Cần lưu ý quan sát  
   - **Suspicious (51–75%)**: Rủi ro đáng kể, có dấu hiệu bất thường  
   - **High Risk (76–100%)**: Cảnh báo độc hại, khuyến nghị ngăn chặn truy cập  

5. **Tích hợp SHAP**:  
   CNN-LSTM: DeepExplainer/GradientExplainer — biểu đồ quy kết theo ký tự URL.  
   XGBoost: TreeExplainer — biểu đồ feature importance và beeswarm summary plot.  
   Xuất toàn bộ kết quả ra `reports/`.

---

## 5. Kết quả Dự kiến

- **Sản phẩm phần mềm**:  
  Ứng dụng web đánh giá rủi ro URL vận hành cục bộ, tích hợp hai model score độc lập và hiển thị kết quả SHAP trực quan.

- **Tính minh bạch dữ liệu**:  
  SHAP TreeExplainer cho XGBoost cung cấp giải thích feature-level tức thì; SHAP DeepExplainer cho CNN-LSTM cung cấp giải thích character-level. Hai góc nhìn bổ sung lẫn nhau.

- **Hiệu năng mô hình**:  
  CNN-LSTM kỳ vọng nắm bắt tốt các mẫu DGA và obfuscation ký tự. XGBoost kỳ vọng mạnh ở các dấu hiệu cấu trúc rõ ràng (IP host, suspicious TLD, keyword density). Cả hai được đánh giá bằng Accuracy, Precision, Recall, F1, ROC-AUC trên tập test độc lập.

- **Sản phẩm báo cáo**:  
  Báo cáo chuyên đề phân tích toàn diện phương pháp luận, so sánh hai mô hình, và luận giải kết quả SHAP.