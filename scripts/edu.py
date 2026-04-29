import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from urllib.parse import urlparse
import numpy as np
import time

# Cấu hình hiển thị
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

def load_data(filepath, sample_frac=0.1):
    """Tải và lấy mẫu dữ liệu để tăng tốc độ EDA trên 10 triệu dòng"""
    print(f"[*] Đang tải dữ liệu từ {filepath}...")
    start = time.time()
    
    # Đọc dữ liệu (có thể dùng engine='c' mặc định để load nhanh)
    df = pd.read_csv(filepath)
    print(f"[+] Đã tải {len(df):,} dòng trong {time.time() - start:.2f}s")
    
    # Lấy mẫu để EDA nhanh hơn (Tùy chọn)
    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=42).copy()
        print(f"[+] Lấy mẫu {sample_frac*100}% -> {len(df):,} dòng để phân tích")
        
    return df

def extract_features(df):
    """Trích xuất đặc trưng Lexical bằng Vectorization cho tốc độ cao"""
    print("[*] Đang trích xuất đặc trưng...")
    
    # Đặc trưng độ dài
    df['url_len'] = df['url'].str.len()
    
    # Số lượng số (digits)
    df['digit_count'] = df['url'].str.count(r'\d')
    df['digit_ratio'] = df['digit_count'] / df['url_len']
    
    # Đếm ký tự đặc biệt
    special_chars = ['-', '@', '?', '=', '.', '_']
    for char in special_chars:
        df[f'count_{char}'] = df['url'].str.count(f'\\{char}' if char in ['?', '.'] else char)
        
    # Trích xuất domain length (Dùng apply nhanh với lambda)
    df['domain'] = df['url'].apply(lambda x: urlparse(x).netloc)
    df['domain_len'] = df['domain'].str.len()
    
    return df

def run_eda(df):
    """Thực hiện trực quan hóa và in báo cáo thống kê"""
    print("\n" + "="*50)
    print("BÁO CÁO THỐNG KÊ CƠ BẢN")
    print("="*50)
    
    # 1. Phân phối nhãn
    print("\n[1] Phân phối nhãn (Label Distribution):")
    print(df['label'].value_counts(normalize=True) * 100)
    
    plt.figure(figsize=(6,4))
    sns.countplot(data=df, x='label', palette='viridis')
    plt.title('Phân phối nhãn (0: Benign, 1: Malicious)')
    plt.savefig('eda_label_dist.png')
    plt.close()

    # 2. Phân phối độ dài URL
    print("\n[2] Thống kê độ dài URL:")
    print(df.groupby('label')['url_len'].describe())
    
    plt.figure()
    sns.histplot(data=df, x='url_len', hue='label', bins=100, kde=True, palette='viridis', log_scale=(False, True))
    plt.xlim(0, 200) # Giới hạn hiển thị để dễ nhìn
    plt.title('Phân phối độ dài URL (Log Scale)')
    plt.savefig('eda_url_length.png')
    plt.close()

    # 3. Phân phối độ dài Domain
    plt.figure()
    sns.boxplot(data=df, x='label', y='domain_len', palette='viridis')
    plt.title('Phân phối độ dài Domain')
    plt.ylim(0, 100)
    plt.savefig('eda_domain_length.png')
    plt.close()

    # 4. Tỷ lệ chữ số trong URL (Digit Ratio)
    plt.figure()
    sns.kdeplot(data=df, x='digit_ratio', hue='label', fill=True, palette='viridis')
    plt.title('Phân phối tỷ lệ chữ số trong URL')
    plt.savefig('eda_digit_ratio.png')
    plt.close()

    # 5. Tần suất ký tự đặc biệt
    chars_to_plot = ['count_-', 'count_?', 'count_=', 'count_.']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for i, char_col in enumerate(chars_to_plot):
        ax = axes[i//2, i%2]
        sns.boxplot(data=df, x='label', y=char_col, ax=ax, palette='viridis', showfliers=False)
        ax.set_title(f'Phân phối {char_col}')
    plt.tight_layout()
    plt.savefig('eda_special_chars.png')
    plt.close()
    
    print("\n[*] Đã lưu các biểu đồ: eda_label_dist.png, eda_url_length.png, eda_domain_length.png, eda_digit_ratio.png, eda_special_chars.png")

if __name__ == "__main__":
    FILE_PATH = "urls_synthetic_10m.csv"
    
    # Load 10% dữ liệu (1 triệu dòng) để EDA chạy mượt mà trên mọi máy local
    df_sample = load_data(FILE_PATH, sample_frac=0.1)
    
    # Trích xuất đặc trưng
    df_features = extract_features(df_sample)
    
    # Chạy EDA & Vẽ biểu đồ
    run_eda(df_features)