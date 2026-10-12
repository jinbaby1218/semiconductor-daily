
import email.utils
import json
import random
import re
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from summarize_papers import add_chinese_briefs

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "papers.json"
API = "https://export.arxiv.org/api/query"

# 最多请求 5 次，包括首次请求。
MAX_ATTEMPTS = 5
REQUEST_TIMEOUT = 45

NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}

CATEGORIES = {
    "lithography": "DUV 光刻",
    "metrology": "量检测设备",
    "packaging": "先进封装",
    "manufacturing": "半导体制造",
    "materials": "材料与工艺",
}


def text_of(parent, path):
    node = parent.find(path, NS)
    if node is None or node.text is None:
        return ""
    return " ".join(node.text.split())


def classify(title, summary):
    value = (title + " " + summary).lower()

    if any(x in value for x in (
        "photolithography", "photoresist", "lithography",
        "lithographic", "patterning"
    )):
        return CATEGORIES["lithography"]

    if any(x in value for x in (
        "metrology", "defect inspection", "critical dimension",
        "cd-sem", "wafer inspection"
    )):
        return CATEGORIES["metrology"]

    if any(x in value for x in (
        "hybrid bonding", "advanced packaging", "chiplet"
    )):
        return CATEGORIES["packaging"]

    if any(x in value for x in (
        "semiconductor", "wafer", "process control",
        "yield", "fabrication"
    )):
        return CATEGORIES["manufacturing"]

    return CATEGORIES["materials"]


def retry_delay(exc, attempt):
    """优先遵循 Retry-After，否则采用指数退避。"""
    retry_after = exc.headers.get("Retry-After") if exc.headers else None

    if retry_after:
        try:
            # Retry-After: 120
            delay = float(retry_after)
        except (TypeError, ValueError):
            try:
                # Retry-After: Wed, 21 Oct 2026 07:28:00 GMT
                retry_time = email.utils.parsedate_to_datetime(retry_after)
                if retry_time.tzinfo is None:
                    retry_time = retry_time.replace(tzinfo=timezone.utc)
                delay = (
                    retry_time - datetime.now(timezone.utc)
                ).total_seconds()
            except (TypeError, ValueError, OverflowError):
                delay = 30 * (2 ** attempt)
    else:
        delay = 30 * (2 ** attempt)

    # 不短于服务器要求的等待时间；避免因随机抖动而提前重试。
    return max(1.0, delay) + random.uniform(0, 3)


def open_with_retry(request, attempts=MAX_ATTEMPTS,
                    timeout=REQUEST_TIMEOUT):
    """有限重试，避免连续请求加重 arXiv 服务端压力。"""
    for attempt in range(attempts):
        try:
            return urllib.request.urlopen(request, timeout=timeout)

        except urllib.error.HTTPError as exc:
            retryable = exc.code == 429 or 500 <= exc.code < 600

            if not retryable or attempt == attempts - 1:
                raise

            delay = retry_delay(exc, attempt)
            print(
                f"arXiv 返回 HTTP {exc.code}；"
                f"{delay:.1f} 秒后重试 "
                f"({attempt + 1}/{attempts - 1})",
                flush=True,
            )
            exc.close()
            time.sleep(delay)

        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if attempt == attempts - 1:
                raise

            delay = min(30 * (2 ** attempt), 300)
            delay += random.uniform(0, 3)
            print(
                f"arXiv 网络请求失败：{exc}；"
                f"{delay:.1f} 秒后重试 "
                f"({attempt + 1}/{attempts - 1})",
                flush=True,
            )
            time.sleep(delay)

    raise RuntimeError("arXiv 请求重试失败")


def fetch():
    query = (
        'all:photolithography OR all:semiconductor OR '
        'all:metrology OR all:photoresist OR all:wafer OR '
        'all:"process control"'
    )

    params = urllib.parse.urlencode({
        "search_query": query,
        "start": 0,
        "max_results": 60,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    })

    request = urllib.request.Request(
        API + "?" + params,
        headers={
            "User-Agent": (
                "SemiconductorDaily/1.0 "
                "(automated academic paper index)"
            ),
            "Accept": "application/atom+xml",
        },
    )

    with open_with_retry(request) as response:
        xml_data = response.read()

    root = ET.fromstring(xml_data)
    articles = []

    for entry in root.findall("atom:entry", NS):
        title = text_of(entry, "atom:title")
        abstract = text_of(entry, "atom:summary")
        paper_url = text_of(entry, "atom:id")

        if not title or not abstract or not paper_url:
            continue

        match = re.search(r"(\d{4}\.\d{4,5})", paper_url)
        arxiv_id = (
            match.group(1)
            if match
            else paper_url.rsplit("/", 1)[-1]
        )

        published = text_of(entry, "atom:published")[:10]

        authors = [
            text_of(author, "atom:name")
            for author in entry.findall("atom:author", NS)
        ]

        pdf_url = "https://arxiv.org/pdf/" + arxiv_id

        for link in entry.findall("atom:link", NS):
            if link.attrib.get("title") == "pdf":
                pdf_url = link.attrib.get("href", pdf_url)
                break

        category = classify(title, abstract)
        combined_text = (title + " " + abstract).lower()
        keywords = [category]

        for word in (
            "DUV", "EUV", "lithography", "metrology",
            "semiconductor", "wafer", "photoresist",
            "defect", "yield", "process control"
        ):
            if word.lower() in combined_text:
                keywords.append(word)

        articles.append({
            "id": "arxiv-" + arxiv_id,
            "arxivId": arxiv_id,
            "category": category,
            "title": title,
            "source": "arXiv",
            "date": published or "Unknown",
            "read": max(2, min(12, len(abstract) // 900 + 2)),
            "keywords": list(dict.fromkeys(keywords))[:5],
            "summary": abstract,
            "detail": (
                "Authors: " + ", ".join(authors)
                + "\n\n" + abstract
            ),
            "url": paper_url,
            "pdfUrl": pdf_url,
            "authors": authors,
            "abstract": abstract,
        })

    if not articles:
        raise RuntimeError(
            "arXiv returned no usable articles; "
            "existing papers.json was preserved"
        )

    # 生成或复用中文速读
    articles = add_chinese_briefs(articles)
    
    payload = {
        "updatedAt": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        ),
        "source": "arXiv API",
        "count": len(articles),
        "articles": articles,
    }

    # 先写入同目录临时文件，写入成功后再原子替换。
    # 即使写入中断，也不会直接截断原有 papers.json。
    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=ROOT,
            prefix=".papers-",
            suffix=".tmp",
            delete=False,
        ) as temp_file:
            temp_path = Path(temp_file.name)
            json.dump(
                payload,
                temp_file,
                ensure_ascii=False,
                indent=2,
            )
            temp_file.write("\n")
            temp_file.flush()

        temp_path.replace(OUTPUT)
        temp_path = None

    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass

    print(
        f"Saved {len(articles)} articles to {OUTPUT}",
        flush=True,
    )


if __name__ == "__main__":
    fetch()
