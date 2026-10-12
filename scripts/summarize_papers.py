
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPERS_FILE = ROOT / "papers.json"

MODEL = "gemini-2.5-flash-lite"
API_URL = (
    "https://generativelanguage.googleapis.com/"
    f"v1beta/models/{MODEL}:generateContent"
)

# 第一次建议设置为 3，成功后再逐步提高。
MAX_NEW_BRIEFS = int(os.getenv("MAX_NEW_BRIEFS", "3"))


def load_old_briefs():
    if not PAPERS_FILE.exists():
        return {}

    data = json.loads(
        PAPERS_FILE.read_text(encoding="utf-8")
    )

    results = {}

    for paper in data.get("articles", []):
        if paper.get("brief_cn"):
            results[paper["id"]] = paper

    return results


def generate_brief(paper, api_key):
    prompt = f"""
你是一名严谨的科技论文编辑。

请依据以下论文标题和英文摘要，
生成一句30至60个汉字的简体中文概括。

要求：
1. 准确说明研究的对象、方法或主要结果。
2. 不编造实验数据或研究结论。
3. 不强行将无关论文解释为半导体制造研究。
4. 不添加标题、引号、序号或其他解释。
5. 只输出一句中文。

英文标题：
{paper["title"]}

英文摘要：
{paper["summary"]}
"""

    body = {
        "contents": [{
            "role": "user",
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 180
        }
    }

    for attempt in range(3):
        request = urllib.request.Request(
            API_URL,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": api_key,
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(
                request, timeout=60
            ) as response:
                data = json.load(response)

            candidates = data.get("candidates", [])
            if not candidates:
                raise RuntimeError("Gemini 未返回摘要")

            parts = (
                candidates[0]
                .get("content", {})
                .get("parts", [])
            )

            brief = "".join(
                part.get("text", "")
                for part in parts
            ).strip()

            brief = " ".join(brief.split())

            if not brief or len(brief) > 180:
                raise RuntimeError("中文摘要格式异常")

            return brief

        except urllib.error.HTTPError as error:
            if error.code not in (429, 500, 502, 503, 504):
                raise

            if attempt == 2:
                raise

            retry_after = error.headers.get("Retry-After")
            try:
                wait = min(
                    max(float(retry_after), 1), 60
                )
            except (ValueError, TypeError):
                wait = 5 * (2 ** attempt)

            print(
                f"Gemini HTTP {error.code}, "
                f"retry in {wait}s",
                flush=True
            )
            time.sleep(wait)

        except (urllib.error.URLError, TimeoutError):
            if attempt == 2:
                raise
            time.sleep(5 * (2 ** attempt))

    raise RuntimeError("Gemini 摘要生成失败")


def add_chinese_briefs(articles):
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "未配置 GEMINI_API_KEY，保留旧 papers.json"
        )

    old_briefs = load_old_briefs()
    generated = 0
    reused = 0

    for paper in articles:
        old = old_briefs.get(paper["id"])

        # 同一篇论文的标题和摘要没变，直接复用。
        if (
            old
            and old.get("title") == paper["title"]
            and old.get("summary") == paper["summary"]
        ):
            paper["brief_cn"] = old["brief_cn"]
            reused += 1
            continue

        # 免费额度保护：每次只生成指定数量。
        if generated >= MAX_NEW_BRIEFS:
            continue

        print(
            f"Generating Chinese brief: {paper['id']}",
            flush=True
        )

        paper["brief_cn"] = generate_brief(
            paper, api_key
        )

        generated += 1
        time.sleep(2)

    print(
        f"Chinese briefs: generated={generated}, "
        f"reused={reused}, "
        f"remaining={len(articles)-generated-reused}",
        flush=True
    )

    return articles
