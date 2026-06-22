"""
Script cào dữ liệu báo chí tiếng Việt mẫu (VnExpress) phục vụ mở rộng tập test-set.
Sử dụng thư viện requests và BeautifulSoup4 để bóc tách tiêu đề và nội dung bài viết.
"""

import os
import json
import re
import unicodedata
import requests
from bs4 import BeautifulSoup

def normalize_vietnamese(text):
    """
    Chuẩn hóa bảng mã Unicode NFC cho tiếng Việt và loại bỏ khoảng trắng thừa.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def scrape_vnexpress_article(url):
    """
    Cào nội dung chi tiết của một bài viết từ VnExpress.
    
    Args:
        url (str): Đường dẫn bài viết VnExpress.
        
    Returns:
        dict: Chứa thông tin tiêu đề, mô tả và nội dung bài viết.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code != 200:
            print(f"Lỗi: Không thể truy cập link {url} (Status code: {response.status_code})")
            return None
            
        soup = BeautifulSoup(response.content, "html.parser")
        
        # Lấy tiêu đề
        title_tag = soup.find("h1", class_="title-detail")
        title = normalize_vietnamese(title_tag.get_text()) if title_tag else ""
        
        # Lấy mô tả ngắn
        desc_tag = soup.find("p", class_="description")
        description = normalize_vietnamese(desc_tag.get_text()) if desc_tag else ""
        
        # Lấy các đoạn nội dung chính
        content_paragraphs = []
        body_tag = soup.find("article", class_="fck_detail")
        if body_tag:
            p_tags = body_tag.find_all("p", class_="Normal")
            for p in p_tags:
                text = normalize_vietnamese(p.get_text())
                if text:
                    content_paragraphs.append(text)
                    
        full_content = " ".join(content_paragraphs)
        
        if not title and not full_content:
            print(f"Cảnh báo: Không trích xuất được nội dung bài viết từ {url}")
            return None
            
        return {
            "url": url,
            "title": title,
            "description": description,
            "content": full_content,
            "total_words": len(full_content.split())
        }
        
    except Exception as e:
        print(f"Lỗi khi cào {url}: {e}")
        return None

def main():
    # URL mẫu tin tức thời sự nóng trên VnExpress
    urls = [
        "https://vnexpress.net/thu-tuong-viet-nam-huong-toi-tu-chu-cong-nghe-va-ai-4750000.html",  # URL giả lập minh họa
        "https://vnexpress.net/xu-huong-phat-trien-vi-mach-ban-dan-tai-viet-nam-4760000.html"      # URL giả lập minh họa
    ]
    
    # Ở đây chúng ta chỉ dùng link thật làm mẫu chạy thử nếu cần thiết
    print("=== SCRAPER TIN TỨC TIẾNG VIỆT MẪU ===")
    
    scraped_data = []
    for url in urls:
        print(f"Đang cào bài viết: {url}...")
        article = scrape_vnexpress_article(url)
        if article:
            print(f"-> Đã cào thành công: {article['title']} ({article['total_words']} từ)")
            scraped_data.append(article)
            
    # Lưu kết quả
    output_dir = "datasets"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "scraped_news_sample_output.json")
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(scraped_data, f, ensure_ascii=False, indent=4)
        
    print(f"\nĐã lưu kết quả cào mẫu vào file: {output_file}")

if __name__ == "__main__":
    main()
