#!/usr/bin/env python3
"""
Wrapper pipeline for ksanjeev284/reddit-universal-scraper.

What it does:
1) Runs the repository scraper in FULL mode on a list of subreddits.
2) Reads posts.csv and comments.csv from data/r_<subreddit>/.
3) Filters out posts unrelated to Claude / Anthropic.
4) Optionally keeps only viral posts based on Reddit engagement thresholds.
5) Bundles each matched post with its comments.
6) Writes combined outputs and topic-specific outputs under data/claude_filtered/.

Run from the root of the cloned reddit-universal-scraper repo.This code is written by AI.

Examples:
    python3 reddit_claude_pipeline.py --limit 250 --no-media
    python3 reddit_claude_pipeline.py --limit 250 --no-media --viral-only
    python3 reddit_claude_pipeline.py --limit 250 --no-media --viral-only --min-score 150 --min-comments 30
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from pathlib import Path

DEFAULT_SUBREDDITS = [
    "claude",
    "ClaudeAI",
    "ClaudeCode",
    "Anthropic",
    "LocalLLaMA",
    "ChatGPT",
    "OpenAI",
    "singularity",
    "MachineLearning",
    "artificial",
    "artificialintelligence",
    "LLMDevs",
    "PromptEngineering",
    "Bard",
    "GeminiAI",
    "compsci",
    "programming",
    "webdev",
    "experienceddevs",
    "SaaS",
    "startups",
    "Futurology",
    "MachineLearningNews",
    "AGI",
]

OUTPUT_ROOT = Path("data") / "claude_filtered"

# Strong Claude / Anthropic indicators
DIRECT_TERMS = [
    r"\bclaude\b",
    r"\bclaude ai\b",
    r"\bclaudeapp\b",
    r"\bclaude\.ai\b",
    r"\bclaude code\b",
    r"\bclaudecode\b",
    r"\bclaude desktop\b",
    r"\bclaude mobile\b",
    r"\bclaude web\b",
    r"\bclaude api\b",
    r"\banthropic\b",
    r"\banthropic api\b",
    r"\banthropic console\b",
    r"\banthropic sdk\b",

    # Model families
    r"\bclaude haiku\b",
    r"\bclaude sonnet\b",
    r"\bclaude opus\b",
    r"\bhaiku 3(?:\.5)?\b",
    r"\bsonnet 3(?:\.5|\.7)?\b",
    r"\bopus 3\b",
    r"\bopus 4(?:\.6)?\b",
    r"\bsonnet 4(?:\.6)?\b",
    r"\bhaiku 4(?:\.5)?\b",
    r"\bclaude 2(?:\.1)?\b",
    r"\bclaude 3\b",
    r"\bclaude 3\.5\b",
    r"\bclaude 3\.7\b",
    r"\bclaude 4\b",
    r"\bclaude 4\.5\b",
    r"\bclaude 4\.6\b",
    r"\bclaude instant\b",

    # Official/near-official feature terms
    r"\bcomputer use\b",
    r"\bartifacts?\b",
    r"\bmcp\b",
    r"\bmodel context protocol\b",
    r"\bremote mcp\b",
    r"\blocal mcp\b",
    r"\btool use\b",
    r"\bfunction calling\b",
    r"\bcode execution\b",
    r"\bextended thinking\b",
    r"\bthinking mode\b",
    r"\bprompt caching\b",
    r"\bbatch api\b",
    r"\bmessages api\b",
    r"\btoken counting\b",
    r"\bcontext window\b",
    r"\bcontext editing\b",
    r"\bworkbench\b",
    r"\bconnectors?\b",
    r"\bextensions?\b",
    r"\bagent sdk\b",
    r"\bclaude agent sdk\b",
    r"\bclaude code sdk\b",
    r"\bconstitutional ai\b",

    # Common usage / product-plan terms
    r"\bclaude pro\b",
    r"\bclaude max\b",
    r"\bclaude team\b",
    r"\bclaude enterprise\b",
    r"\bmax plan\b",
    r"\bsonnet thinking\b",
    r"\bopus thinking\b",
    r"\bhaiku thinking\b",

    # Ecosystem / community wording
    r"\bclaude projects?\b",
    r"\bclaude skills?\b",
    r"\bclaude connectors?\b",
    r"\bclaude extensions?\b",
    r"\bclaude agents?\b",
    r"\busing claude\b",
    r"\bbuilt with claude\b",
    r"\bpowered by claude\b",

    # Useful loose variants / typos
    r"\banthropic'?s\b",
    r"\bclaud[e]?\s*code\b",
    r"\bclaud[e]?\s*ai\b",
]

# Competitor / comparison vocabulary
# IMPORTANT: do not use these alone as the relevance decision.
# Best practice: require DIRECT_TERMS match OR strong Anthropic subreddit context.
COMPARISON_TERMS = [
    r"\bopenai\b",
    r"\bchatgpt\b",
    r"\bgpt\b",
    r"\bgpt-?4\b",
    r"\bgpt-?4\.1\b",
    r"\bgpt-?4o\b",
    r"\bgpt-?5\b",
    r"\bo1\b",
    r"\bo3\b",
    r"\bo4-?mini\b",
    r"\bcodex\b",

    r"\bgemini\b",
    r"\bgoogle ai\b",
    r"\bgoogle deepmind\b",
    r"\bdeepmind\b",
    r"\bgemini 1\.5\b",
    r"\bgemini 2(?:\.0|\.5)?\b",

    r"\bdeepseek\b",
    r"\bdeepseek[- ]?r1\b",
    r"\bdeepseek[- ]?v3\b",

    r"\bgrok\b",
    r"\bxai\b",

    r"\bllama\b",
    r"\blocal llama\b",
    r"\bmeta ai\b",
    r"\bmeta llama\b",

    r"\bmistral\b",
    r"\bmixtral\b",

    r"\bqwen\b",
    r"\balibaba qwen\b",

    r"\bperplexity\b",
    r"\bmanus\b",
    r"\bdevin\b",
    r"\bcursor\b",
    r"\bwindsurf\b",
    r"\bgithub copilot\b",
    r"\bcopilot\b",
    r"\breplit agent\b",
    r"\bbolt\.new\b",
    r"\blovable\b",
    r"\bphind\b",
    r"\byou\.com\b",
    r"\bpoe\b",
]

CLAUDE_DIRECT_RE = re.compile("|".join(DIRECT_TERMS), re.IGNORECASE)
COMPARISON_RE = re.compile("|".join(COMPARISON_TERMS), re.IGNORECASE)

# Model names alone are too broad, so this stricter pattern requires Claude/Anthropic context.
MODEL_WITH_CONTEXT_RE = re.compile(
    r"(?:claude|anthropic).{0,40}(?:sonnet|opus|haiku|3(?:\.5|\.7)?|4(?:\.|\b))|"
    r"(?:sonnet|opus|haiku|3(?:\.5|\.7)?|4(?:\.|\b)).{0,40}(?:claude|anthropic)",
    re.IGNORECASE | re.DOTALL,
    )

CLAUDE_CODE_RE = re.compile(
    r"\bclaude code\b|\bcoding agent\b|\bterminal\b|\brepo\b|\bpull request\b|\bcode review\b",
    re.IGNORECASE,
)
ANTHROPIC_COMPANY_RE = re.compile(
    r"\banthropic\b|\bapi\b|\bpricing\b|\bsafety\b|\bpolicy\b|\bcompany\b|\benterprise\b",
    re.IGNORECASE,
)


@dataclass
class ScrapeResult:
    subreddit: str
    ok: bool
    message: str


@dataclass
class ViralConfig:
    enabled: bool
    min_score: int
    min_comments: int
    min_awards: int
    min_upvote_ratio: float


def normalize_permalink(value: object) -> str:
    if value is None:
        return ""
    s = str(value).strip()
    if not s:
        return ""
    if s.endswith(".json"):
        s = s[:-5]
    if s.endswith("/"):
        s = s[:-1]
    return s



def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except UnicodeDecodeError:
        return pd.read_csv(path, low_memory=False, encoding="latin-1")



def run_scrape(repo_root: Path, subreddit: str, limit: int, include_media: bool, include_plugins: bool) -> ScrapeResult:
    cmd = [
        sys.executable,
        "main.py",
        subreddit,
        "--mode",
        "full",
        "--limit",
        str(limit),
    ]
    if not include_media:
        cmd.append("--no-media")
    if include_plugins:
        cmd.append("--plugins")

    print(f"\n=== Scraping r/{subreddit} ===")
    print("$", " ".join(cmd))

    proc = subprocess.run(cmd, cwd=repo_root, text=True)
    if proc.returncode == 0:
        return ScrapeResult(subreddit, True, "completed")
    return ScrapeResult(subreddit, False, f"scraper exited with code {proc.returncode}")



def build_post_text(row: pd.Series) -> str:
    parts = [
        str(row.get("title", "") or ""),
        str(row.get("selftext", "") or ""),
        str(row.get("flair", "") or ""),
        str(row.get("url", "") or ""),
    ]
    return "\n".join(parts)



def build_comment_text(df: pd.DataFrame) -> str:
    if df.empty or "body" not in df.columns:
        return ""
    return "\n".join(df["body"].fillna("").astype(str).tolist())



def is_related(post_text: str, comment_text: str) -> Tuple[bool, str]:
    merged = f"{post_text}\n{comment_text}"
    merged_lower = merged.lower()

    post_has_direct = bool(CLAUDE_DIRECT_RE.search(post_text)) or bool(MODEL_WITH_CONTEXT_RE.search(post_text))
    comments_have_direct = bool(CLAUDE_DIRECT_RE.search(comment_text)) or bool(MODEL_WITH_CONTEXT_RE.search(comment_text))

    if post_has_direct:
        return True, "direct_post_match"
    if comments_have_direct:
        return True, "comment_match"

    post_has_comparison = bool(COMPARISON_RE.search(post_text))
    comments_have_comparison = bool(COMPARISON_RE.search(comment_text))
    if (post_has_comparison or comments_have_comparison) and ("claude" in merged_lower or "anthropic" in merged_lower):
        return True, "comparison_context"

    return False, "filtered_out"



def classify_topic(post_text: str, comment_text: str) -> str:
    merged = f"{post_text}\n{comment_text}"
    merged_lower = merged.lower()
    has_comparison = bool(COMPARISON_RE.search(merged)) and ("claude" in merged_lower or "anthropic" in merged_lower)

    if CLAUDE_CODE_RE.search(merged):
        return "claude_code"
    if has_comparison:
        return "model_comparison"
    if MODEL_WITH_CONTEXT_RE.search(merged):
        return "anthropic_models"
    if ANTHROPIC_COMPANY_RE.search(merged):
        return "anthropic_company"
    return "core_claude"



def to_json_safe(value):
    if pd.isna(value):
        return None
    return value



def save_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)



def save_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)



def as_int(value: object, default: int = 0) -> int:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except TypeError:
        pass
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default



def as_float(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        if pd.isna(value):
            return default
    except TypeError:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return default



def compute_viral_score(post: pd.Series) -> float:
    score = as_int(post.get("score", 0))
    comments = as_int(post.get("num_comments", 0))
    crossposts = as_int(post.get("num_crossposts", 0))
    awards = as_int(post.get("total_awards", 0))
    upvote_ratio = as_float(post.get("upvote_ratio", 0.0))
    return float(score + 2 * comments + 10 * crossposts + 15 * awards + 50 * upvote_ratio)



def is_viral(post: pd.Series, cfg: ViralConfig) -> bool:
    if not cfg.enabled:
        return True

    score = as_int(post.get("score", 0))
    num_comments = as_int(post.get("num_comments", 0))
    total_awards = as_int(post.get("total_awards", 0))
    upvote_ratio = as_float(post.get("upvote_ratio", 0.0))

    return (
            score >= cfg.min_score
            and num_comments >= cfg.min_comments
            and total_awards >= cfg.min_awards
            and upvote_ratio >= cfg.min_upvote_ratio
    )



def enrich_post_metrics(post: pd.Series) -> dict:
    score = as_int(post.get("score", 0))
    num_comments = as_int(post.get("num_comments", 0))
    total_awards = as_int(post.get("total_awards", 0))
    num_crossposts = as_int(post.get("num_crossposts", 0))
    upvote_ratio = as_float(post.get("upvote_ratio", 0.0))
    viral_score = compute_viral_score(post)
    return {
        "score": score,
        "num_comments": num_comments,
        "total_awards": total_awards,
        "num_crossposts": num_crossposts,
        "upvote_ratio": upvote_ratio,
        "viral_score": viral_score,
    }



def process_subreddit(repo_root: Path, subreddit: str, viral_cfg: ViralConfig) -> Dict[str, object]:
    raw_dir = repo_root / "data" / f"r_{subreddit}"
    posts_path = raw_dir / "posts.csv"
    comments_path = raw_dir / "comments.csv"

    posts_df = safe_read_csv(posts_path)
    comments_df = safe_read_csv(comments_path)

    if posts_df.empty:
        print(f"No posts found for r/{subreddit} at {posts_path}")
        return {
            "subreddit": subreddit,
            "total_posts": 0,
            "matched_posts": 0,
            "matched_comments": 0,
            "viral_kept_posts": 0,
        }

    if "permalink" not in posts_df.columns:
        posts_df["permalink"] = ""
    posts_df["normalized_permalink"] = posts_df["permalink"].apply(normalize_permalink)
    posts_df["source_subreddit"] = subreddit

    if comments_df.empty:
        comments_df = pd.DataFrame(columns=["post_permalink", "body"])
    if "post_permalink" not in comments_df.columns:
        comments_df["post_permalink"] = ""
    if "body" not in comments_df.columns:
        comments_df["body"] = ""
    comments_df["normalized_post_permalink"] = comments_df["post_permalink"].apply(normalize_permalink)
    comments_df["source_subreddit"] = subreddit

    grouped_comments = {
        key: grp.copy()
        for key, grp in comments_df.groupby("normalized_post_permalink", dropna=False)
    }

    matched_posts_records: List[dict] = []
    matched_comments_frames: List[pd.DataFrame] = []
    bundled: List[dict] = []
    topic_bundles: Dict[str, List[dict]] = {}
    relevance_kept_count = 0

    for _, post in posts_df.iterrows():
        normalized_permalink = post.get("normalized_permalink", "")
        related_comments = grouped_comments.get(normalized_permalink, pd.DataFrame()).copy()

        post_text = build_post_text(post)
        comment_text = build_comment_text(related_comments)

        keep, reason = is_related(post_text, comment_text)
        if not keep:
            continue

        relevance_kept_count += 1

        if not is_viral(post, viral_cfg):
            continue

        topic = classify_topic(post_text, comment_text)
        metrics = enrich_post_metrics(post)

        post_dict = {
            k: to_json_safe(v)
            for k, v in post.drop(labels=["normalized_permalink"], errors="ignore").to_dict().items()
        }
        post_dict.update(metrics)
        post_dict["filter_reason"] = reason
        post_dict["topic"] = topic
        post_dict["comment_count_filtered_bundle"] = int(len(related_comments))
        post_dict["viral_only_mode"] = viral_cfg.enabled
        matched_posts_records.append(post_dict)

        if not related_comments.empty:
            temp_comments = related_comments.copy()
            temp_comments["topic"] = topic
            temp_comments["parent_post_topic"] = topic
            temp_comments["parent_post_viral_score"] = metrics["viral_score"]
            temp_comments["parent_post_score"] = metrics["score"]
            temp_comments["parent_post_num_comments"] = metrics["num_comments"]
            matched_comments_frames.append(temp_comments)
            comment_records = [
                {k: to_json_safe(v) for k, v in rec.items()}
                for rec in temp_comments.drop(columns=["normalized_post_permalink"], errors="ignore").to_dict(orient="records")
            ]
        else:
            comment_records = []

        bundle_item = dict(post_dict)
        bundle_item["comments"] = comment_records
        bundled.append(bundle_item)
        topic_bundles.setdefault(topic, []).append(bundle_item)

    matched_posts_df = pd.DataFrame(matched_posts_records)
    matched_comments_df = pd.concat(matched_comments_frames, ignore_index=True) if matched_comments_frames else pd.DataFrame()

    if not matched_posts_df.empty:
        matched_posts_df = matched_posts_df.sort_values(
            by=["viral_score", "score", "num_comments"],
            ascending=[False, False, False],
            na_position="last",
        ).reset_index(drop=True)

    sub_out = OUTPUT_ROOT / "by_subreddit" / subreddit
    save_csv(sub_out / "matched_posts.csv", matched_posts_df)
    save_csv(sub_out / "matched_comments.csv", matched_comments_df)
    save_json(sub_out / "bundled_posts_with_comments.json", bundled)

    for topic, items in topic_bundles.items():
        items_sorted = sorted(
            items,
            key=lambda x: (
                -(as_float(x.get("viral_score", 0.0))),
                -(as_int(x.get("score", 0))),
                -(as_int(x.get("num_comments", 0))),
            ),
        )
        topic_dir = OUTPUT_ROOT / "by_topic" / topic
        save_json(topic_dir / f"{subreddit}.json", items_sorted)

    print(
        f"r/{subreddit}: relevance-kept {relevance_kept_count} / {len(posts_df)} posts; "
        f"final-kept {len(matched_posts_df)} posts and {len(matched_comments_df)} comments"
    )

    return {
        "subreddit": subreddit,
        "total_posts": int(len(posts_df)),
        "relevance_kept_posts": int(relevance_kept_count),
        "matched_posts": int(len(matched_posts_df)),
        "matched_comments": int(len(matched_comments_df)),
        "viral_kept_posts": int(len(matched_posts_df)),
    }



def combine_outputs(summaries: List[Dict[str, object]], viral_cfg: ViralConfig) -> None:
    by_subreddit_dir = OUTPUT_ROOT / "by_subreddit"
    all_posts: List[pd.DataFrame] = []
    all_comments: List[pd.DataFrame] = []
    all_bundle: List[dict] = []

    for sub_dir in sorted(by_subreddit_dir.glob("*")):
        if not sub_dir.is_dir():
            continue
        posts_path = sub_dir / "matched_posts.csv"
        comments_path = sub_dir / "matched_comments.csv"
        bundle_path = sub_dir / "bundled_posts_with_comments.json"

        if posts_path.exists():
            df_posts = safe_read_csv(posts_path)
            if not df_posts.empty:
                all_posts.append(df_posts)
        if comments_path.exists():
            df_comments = safe_read_csv(comments_path)
            if not df_comments.empty:
                all_comments.append(df_comments)
        if bundle_path.exists():
            with bundle_path.open("r", encoding="utf-8") as f:
                all_bundle.extend(json.load(f))

    combined_posts = pd.concat(all_posts, ignore_index=True) if all_posts else pd.DataFrame()
    combined_comments = pd.concat(all_comments, ignore_index=True) if all_comments else pd.DataFrame()

    if not combined_posts.empty:
        combined_posts = combined_posts.sort_values(
            by=["viral_score", "score", "num_comments"],
            ascending=[False, False, False],
            na_position="last",
        ).reset_index(drop=True)

    all_bundle = sorted(
        all_bundle,
        key=lambda x: (
            -(as_float(x.get("viral_score", 0.0))),
            -(as_int(x.get("score", 0))),
            -(as_int(x.get("num_comments", 0))),
        ),
    )

    save_csv(OUTPUT_ROOT / "all_matched_posts.csv", combined_posts)
    save_csv(OUTPUT_ROOT / "all_matched_comments.csv", combined_comments)
    save_json(OUTPUT_ROOT / "all_bundled_posts_with_comments.json", all_bundle)

    topic_counts = {}
    subreddit_counts = {}
    if not combined_posts.empty:
        if "topic" in combined_posts.columns:
            topic_counts = combined_posts["topic"].value_counts(dropna=False).to_dict()
        if "source_subreddit" in combined_posts.columns:
            subreddit_counts = combined_posts["source_subreddit"].value_counts(dropna=False).to_dict()

    summary = {
        "subreddit_summaries": summaries,
        "total_matched_posts": int(len(combined_posts)),
        "total_matched_comments": int(len(combined_comments)),
        "topic_counts": topic_counts,
        "subreddit_counts": subreddit_counts,
        "viral_filter": {
            "enabled": viral_cfg.enabled,
            "min_score": viral_cfg.min_score,
            "min_comments": viral_cfg.min_comments,
            "min_awards": viral_cfg.min_awards,
            "min_upvote_ratio": viral_cfg.min_upvote_ratio,
        },
        "output_root": str(OUTPUT_ROOT),
    }
    save_json(OUTPUT_ROOT / "summary.json", summary)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrape and filter Claude-related Reddit data using reddit-universal-scraper."
    )
    parser.add_argument(
        "--repo-root",
        default=".",
        help="Path to cloned reddit-universal-scraper repo root. Default: current directory",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Per-subreddit post limit for the underlying scraper.",
    )
    parser.add_argument(
        "--subreddits",
        nargs="+",
        default=DEFAULT_SUBREDDITS,
        help="Subreddits to scrape. Default is a broad Claude/AI set.",
    )
    parser.add_argument(
        "--no-media",
        action="store_true",
        help="Skip media downloads in the underlying scraper to make runs faster.",
    )
    parser.add_argument(
        "--no-plugins",
        action="store_true",
        help="Disable repo plugins in the underlying scraper.",
    )
    parser.add_argument(
        "--skip-scrape",
        action="store_true",
        help="Do not run new scrapes; only process existing data/r_<subreddit>/ files.",
    )
    parser.add_argument(
        "--viral-only",
        action="store_true",
        help="Keep only viral Claude-related posts after relevance filtering.",
    )
    parser.add_argument(
        "--min-score",
        type=int,
        default=100,
        help="Minimum Reddit score for viral filter. Default: 100",
    )
    parser.add_argument(
        "--min-comments",
        type=int,
        default=20,
        help="Minimum comment count for viral filter. Default: 20",
    )
    parser.add_argument(
        "--min-awards",
        type=int,
        default=0,
        help="Minimum total awards for viral filter. Default: 0",
    )
    parser.add_argument(
        "--min-upvote-ratio",
        type=float,
        default=0.0,
        help="Minimum upvote ratio for viral filter. Default: 0.0",
    )
    return parser.parse_args()



def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    main_py = repo_root / "main.py"
    if not main_py.exists():
        print(f"Could not find main.py at {main_py}")
        print("Run this script from the root of the cloned reddit-universal-scraper repo, or pass --repo-root.")
        return 1

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    viral_cfg = ViralConfig(
        enabled=args.viral_only,
        min_score=args.min_score,
        min_comments=args.min_comments,
        min_awards=args.min_awards,
        min_upvote_ratio=args.min_upvote_ratio,
    )

    scrape_summaries: List[Dict[str, object]] = []
    if not args.skip_scrape:
        for subreddit in args.subreddits:
            result = run_scrape(
                repo_root=repo_root,
                subreddit=subreddit,
                limit=args.limit,
                include_media=not args.no_media,
                include_plugins=not args.no_plugins,
            )
            if not result.ok:
                print(f"WARNING: r/{subreddit} scrape issue: {result.message}")

    for subreddit in args.subreddits:
        summary = process_subreddit(repo_root, subreddit, viral_cfg)
        scrape_summaries.append(summary)

    combine_outputs(scrape_summaries, viral_cfg)

    print("\nDone.")
    print(f"Filtered outputs written to: {OUTPUT_ROOT}")
    print(f"Combined posts: {OUTPUT_ROOT / 'all_matched_posts.csv'}")
    print(f"Combined comments: {OUTPUT_ROOT / 'all_matched_comments.csv'}")
    print(f"Bundled JSON: {OUTPUT_ROOT / 'all_bundled_posts_with_comments.json'}")
    print(f"Summary: {OUTPUT_ROOT / 'summary.json'}")
    if viral_cfg.enabled:
        print(
            "Viral filter enabled with thresholds: "
            f"score>={viral_cfg.min_score}, comments>={viral_cfg.min_comments}, "
            f"awards>={viral_cfg.min_awards}, upvote_ratio>={viral_cfg.min_upvote_ratio}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
