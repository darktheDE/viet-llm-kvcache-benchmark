import argparse
import hashlib
import json
import os
import re
import time
import unicodedata
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional

import feedparser
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

try:
    import trafilatura
except ImportError:
    trafilatura = None


OUTPUT_DIR = "datasets/expansion"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "raw_expansion_candidates.jsonl")

USER_AGENT = (
    "viet-llm-kvcache-benchmark/0.1 "
    "(academic data curation; contact: team-data)"
)

NEWS_FEEDS = {
    # RSS chính thống, phù hợp để lấy title/summary/url/date.
    "tuoitre_latest": "https://tuoitre.vn/rss/tin-moi-nhat.rss",
    "tuoitre_thoi_su": "https://tuoitre.vn/rss/thoi-su.rss",
    "tuoitre_cong_nghe": "https://tuoitre.vn/rss/cong-nghe.rss",
    "tuoitre_giao_duc": "https://tuoitre.vn/rss/giao-duc.rss",
    "tuoitre_suc_khoe": "https://tuoitre.vn/rss/suc-khoe.rss",

    "vov_thoi_su": "https://vov.gov.vn/rss/thoi-su.rss",
    "vov_kinh_te": "https://vov.gov.vn/rss/kinh-te.rss",
    "vov_khoa_hoc_cong_nghe": "https://vov.gov.vn/rss/khoa-hoc-cong-nghe.rss",

    # Các nguồn dưới đây nên ưu tiên metadata / summary, không public lại full text nếu chưa xin phép.
    "dantri_latest": "https://dantri.com.vn/rss/tin-moi-nhat.rss",
    "dantri_cong_nghe": "https://dantri.com.vn/rss/cong-nghe.rss",
    "dantri_giao_duc": "https://dantri.com.vn/rss/giao-duc.rss",
    "dantri_suc_khoe": "https://dantri.com.vn/rss/suc-khoe.rss",

    "vietnamplus_home": "https://www.vietnamplus.vn/rss/home.rss",
    "vietnamplus_cong_nghe": "https://www.vietnamplus.vn/rss/cong-nghe.rss",
    "vietnamplus_xa_hoi": "https://www.vietnamplus.vn/rss/xa-hoi.rss",
}

GOOGLE_TRENDS_RSS = "https://trends.google.com/trending/rss?geo=VN"

WIKISOURCE_PAGES = [
    "Truyện Kiều",
    "Lục Vân Tiên",
    "Cung oán ngâm khúc",
    "Bình Ngô đại cáo",
    "Hịch tướng sĩ",
]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_id(prefix: str, text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:16]
    return f"{prefix}_{digest}"


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_html(raw_html: str) -> str:
    soup = BeautifulSoup(raw_html or "", "html.parser")

    for tag in soup(["script", "style", "noscript", "iframe", "form", "nav", "footer", "header"]):
        tag.decompose()

    text = soup.get_text(" ")
    return normalize_text(text)


def is_valid_text(text: str, min_chars: int = 300) -> bool:
    if not text or len(text) < min_chars:
        return False
    if "�" in text:
        return False

    # Loại text quá nhiều ký tự rác.
    weird_chars = len(re.findall(r"[^0-9A-Za-zÀ-ỹ\s.,;:!?()\"'/-]", text))
    if weird_chars / max(len(text), 1) > 0.08:
        return False

    return True


def fetch_url(url: str, timeout: int = 20) -> Optional[str]:
    try:
        resp = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=timeout,
        )
        if resp.status_code != 200:
            return None
        return resp.text
    except Exception:
        return None


def extract_article_text(url: str, fallback_html: str = "") -> str:
    html = fetch_url(url)
    if not html:
        html = fallback_html

    if not html:
        return ""

    if trafilatura is not None:
        extracted = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=False,
            favor_precision=True,
        )
        if extracted:
            return normalize_text(extracted)

    return clean_html(html)


def collect_news(max_items: int) -> List[Dict]:
    records = []

    for source_name, feed_url in NEWS_FEEDS.items():
        feed = feedparser.parse(feed_url)

        for entry in feed.entries[:max_items]:
            title = normalize_text(entry.get("title", ""))
            summary = clean_html(entry.get("summary", ""))
            link = entry.get("link", "")
            published = entry.get("published", "") or entry.get("updated", "")

            # Full text chỉ dùng làm candidate nội bộ. Khi public repo, cần kiểm tra quyền nguồn.
            article_text = extract_article_text(link)
            if not is_valid_text(article_text, min_chars=500):
                article_text = f"{title}\n\n{summary}".strip()

            text = normalize_text(article_text)
            if not is_valid_text(text, min_chars=200):
                continue

            records.append({
                "sample_id": stable_id("news", source_name + link + title),
                "source_type": "news",
                "source_name": source_name,
                "url": link,
                "title": title,
                "published_date": published,
                "category": infer_category(source_name),
                "text": text,
                "license_note": "RSS/source metadata collected for academic benchmarking; verify redistribution rights before public release.",
                "collected_at": now_iso(),
            })

            time.sleep(0.5)

    return records


def infer_category(source_name: str) -> str:
    name = source_name.lower()
    if "cong_nghe" in name or "khoa_hoc" in name:
        return "technology"
    if "giao_duc" in name:
        return "education"
    if "suc_khoe" in name or "y_te" in name:
        return "health"
    if "kinh_te" in name:
        return "economy"
    if "thoi_su" in name or "xa_hoi" in name:
        return "society"
    return "general_news"


def collect_google_trends(max_items: int) -> List[Dict]:
    records = []
    feed = feedparser.parse(GOOGLE_TRENDS_RSS)

    for entry in feed.entries[:max_items]:
        title = normalize_text(entry.get("title", ""))
        summary = clean_html(entry.get("summary", ""))
        link = entry.get("link", "")

        text = normalize_text(f"Xu hướng tìm kiếm tại Việt Nam: {title}. {summary}")
        if not is_valid_text(text, min_chars=30):
            continue

        records.append({
            "sample_id": stable_id("trend", title + link),
            "source_type": "trend",
            "source_name": "google_trends_vn",
            "url": link,
            "title": title,
            "published_date": entry.get("published", "") or entry.get("updated", ""),
            "category": "current_trend",
            "text": text,
            "license_note": "Trend metadata only; use for topic discovery and retrieval prompts.",
            "collected_at": now_iso(),
        })

    return records


def collect_wikisource_pages() -> List[Dict]:
    records = []
    api_url = "https://vi.wikisource.org/w/api.php"

    for page in tqdm(WIKISOURCE_PAGES, desc="Wikisource"):
        params = {
            "action": "parse",
            "page": page,
            "prop": "text",
            "format": "json",
            "formatversion": "2",
        }

        try:
            resp = requests.get(
                api_url,
                params=params,
                headers={"User-Agent": USER_AGENT},
                timeout=30,
            )
            data = resp.json()
            html = data.get("parse", {}).get("text", "")
        except Exception:
            continue

        text = clean_html(html)
        if not is_valid_text(text, min_chars=1000):
            continue

        records.append({
            "sample_id": stable_id("book", page + text[:200]),
            "source_type": "book_open",
            "source_name": "vi_wikisource",
            "url": f"https://vi.wikisource.org/wiki/{page.replace(' ', '_')}",
            "title": page,
            "published_date": "",
            "category": "literature_or_historical_text",
            "text": text,
            "license_note": "Wikisource open content; keep attribution and source URL.",
            "collected_at": now_iso(),
        })

        time.sleep(0.5)

    return records


def collect_youtube_trending(max_items: int) -> List[Dict]:
    api_key = os.environ.get("YOUTUBE_API_KEY", "").strip()
    if not api_key:
        return []

    url = "https://www.googleapis.com/youtube/v3/videos"
    params = {
        "part": "snippet,statistics",
        "chart": "mostPopular",
        "regionCode": "VN",
        "maxResults": min(max_items, 50),
        "key": api_key,
    }

    try:
        resp = requests.get(url, params=params, timeout=30)
        data = resp.json()
    except Exception:
        return []

    records = []
    for item in data.get("items", []):
        snippet = item.get("snippet", {})
        stats = item.get("statistics", {})
        video_id = item.get("id", "")

        title = normalize_text(snippet.get("title", ""))
        description = normalize_text(snippet.get("description", ""))
        tags = snippet.get("tags", [])

        text = normalize_text(
            f"Tiêu đề video thịnh hành: {title}. "
            f"Mô tả: {description}. "
            f"Tags: {', '.join(tags[:20])}."
        )

        if not is_valid_text(text, min_chars=50):
            continue

        records.append({
            "sample_id": stable_id("youtube", video_id + title),
            "source_type": "social_metadata",
            "source_name": "youtube_most_popular_vn",
            "url": f"https://www.youtube.com/watch?v={video_id}",
            "title": title,
            "published_date": snippet.get("publishedAt", ""),
            "category": "youtube_trending",
            "text": text,
            "metadata": {
                "channel_title": snippet.get("channelTitle", ""),
                "category_id": snippet.get("categoryId", ""),
                "view_count": stats.get("viewCount", ""),
                "like_count": stats.get("likeCount", ""),
                "comment_count": stats.get("commentCount", ""),
            },
            "license_note": "YouTube metadata only; do not scrape private data or comments without review.",
            "collected_at": now_iso(),
        })

    return records


def deduplicate(records: Iterable[Dict]) -> List[Dict]:
    seen = set()
    output = []

    for record in records:
        key = hashlib.sha1(record["text"].encode("utf-8", errors="ignore")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        output.append(record)

    return output


def write_jsonl(records: List[Dict], output_file: str) -> None:
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-news", type=int, default=20)
    parser.add_argument("--max-trends", type=int, default=50)
    parser.add_argument("--max-youtube", type=int, default=25)
    parser.add_argument("--output", default=OUTPUT_FILE)
    args = parser.parse_args()

    all_records = []

    print("Collecting Wikisource books...")
    all_records.extend(collect_wikisource_pages())

    print("Collecting news RSS...")
    all_records.extend(collect_news(max_items=args.max_news))

    print("Collecting Google Trends...")
    all_records.extend(collect_google_trends(max_items=args.max_trends))

    print("Collecting YouTube trending metadata...")
    all_records.extend(collect_youtube_trending(max_items=args.max_youtube))

    all_records = deduplicate(all_records)
    write_jsonl(all_records, args.output)

    print(f"Saved: {args.output}")
    print(f"Total records: {len(all_records)}")

    by_type = {}
    for r in all_records:
        by_type[r["source_type"]] = by_type.get(r["source_type"], 0) + 1
    print("By source_type:", by_type)


if __name__ == "__main__":
    main()