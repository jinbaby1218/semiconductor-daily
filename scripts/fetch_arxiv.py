
import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "papers.json"
API = "https://export.arxiv.org/api/query"

NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}

CATEGORIES = {
    "lithography": "\u0044\u0055\u0056 \u5149\u523b",
    "metrology": "\u91cf\u68c0\u6d4b\u8bbe\u5907",
    "packaging": "\u5148\u8fdb\u5c01\u88c5",
    "manufacturing": "\u534a\u5bfc\u4f53\u5236\u9020",
    "materials": "\u6750\u6599\u4e0e\u5de5\u827a",
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
        headers={"User-Agent": "SemiconductorDaily/1.0"},
    )

    with urllib.request.urlopen(request, timeout=45) as response:
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
        arxiv_id = match.group(1) if match else paper_url.rsplit("/", 1)[-1]

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
        keywords = [category]

        for word in (
            "DUV", "EUV", "lithography", "metrology",
            "semiconductor", "wafer", "photoresist",
            "defect", "yield", "process control"
        ):
            if word.lower() in (title + " " + abstract).lower():
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
            "detail": "Authors: " + ", ".join(authors) + "\n\n" + abstract,
            "url": paper_url,
            "pdfUrl": pdf_url,
            "authors": authors,
            "abstract": abstract,
        })

    if not articles:
        raise RuntimeError("arXiv returned no usable articles")

    payload = {
        "updatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": "arXiv API",
        "count": len(articles),
        "articles": articles,
    }

    OUTPUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Saved", len(articles), "articles to", OUTPUT)


if __name__ == "__main__":
    fetch()
