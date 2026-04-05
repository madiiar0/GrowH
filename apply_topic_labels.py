from pathlib import Path
import pandas as pd

ARTIFACTS_DIR = Path("artifacts")

POSTS_PATH = ARTIFACTS_DIR / "posts_clustered_clean.csv"
COMMENTS_PATH = Path("all_comments.csv")

POSTS_OUT = ARTIFACTS_DIR / "posts_labeled.csv"
COMMENTS_OUT = ARTIFACTS_DIR / "comments_labeled.csv"

CLUSTER_LABELS = {
    71: "General Claude / AI discussion",
    35: "Claude Code / coding workflow discussion",
    45: "ChatGPT image / Reddit meta",
    141: "Reddit preview-image URL noise",
    106: "ChatGPT image / Reddit meta",
    20: "ChatGPT image / Reddit meta",
    95: "Claude Desktop / connectors outage",
    40: "llama.cpp / local inference",
    55: "CC Lens / npm tool issue",
    59: "AI startup / product idea discussion",
    27: "ML research engineer career discussion",
    11: "Computer Use / Claude Code usage limits",
    130: "Tree-sitter / AST / agent debugging",
    7: "Session limits / peak hours",
    78: "Pro plan / peak-hours complaints",
    124: "Claude memory / context cleanup",
    28: "Claude Desktop / remote control / MCP",
    17: "Claude Code source leak / hidden features",
    81: "Usage-limit feedback / open letter",
    69: "Weekly / session limits",
    21: "Off-topic / noise",
    68: "AMP / scraped URL noise",
    58: "Hobby project / LMS build",
    42: "Token drain / Codex / leaked-source discussion",

    103: "Claude Code defaults / config issue",
    104: "Off-topic / unclear",
    105: "General AI ranking / market talk",
    107: "claude.md / md-file discussion",
    108: "Multi-agent coding discussion",
    109: "claude.md / Spanish-language post",
    102: "Account creation / invalid phone number",
    97: "Grok / Gemini / benchmark comparison",
    101: "Local RAG / MCP / IDE integration",
    100: "Image / preview / extension noise",
    99: "AI ethics / violence discussion",
    98: "ChatGPT image / prompt tricking",
    111: "AI slop / image-generation quality",
    96: "Claude Code leak discussion",
    94: "Google Lens / file-access issue",
    93: "PhD / thesis / graduation off-topic",
    92: "Off-topic personal reflection",
    91: "Malicious / legit AI check",
    90: "Skills / tool invocation discussion",
    89: "Prompt caching / token accounting",
    88: "GPU / local hardware for AI",
    87: "AI market / business strategy",
    86: "General AI productivity / lifestyle",
    110: "AGI / ASI / singularity discussion",
    0: "Agent framework / Autonet",
    112: "Subagents / agent bugs / reward hacking",

    140: "Anthropic transparency / open-source criticism",
    134: "Claude connectors / integrations demo",
    135: "Claude Code plugin / security / CVE",
    136: "Claude bot / platform posting",
    137: "ChatGPT Teams / workspace issue",
    138: "ChatGPT / bash / disguised AI-writing discussion",
    139: "Claude prototype / tool-calling demo",
    142: "Medical imaging paper / preprint",
    132: "Free-user / Sonnet / usage limits",
    143: "Vibe-coded apps / misc",
    144: "Claude Code execution traces / self-improvement",
    145: "Local-model hardware / Mac vs PC",
    146: "Code / user-type / verification test",
    147: "Adult-content / moderation / false-positive",
    148: "AI courses / learning resources",
    133: "Off-topic casual chat",
    131: "AI psychology / internal knowledge systems",

    113: "Google / Gemini / Maps integration",
    119: "Claude account suspended / banned",
    114: "Agent file access / openclaw / codex",
    115: "Claude / Codex / outages / usage gone",
    116: "Data-center / connection / paid-plan issue",
    84: "Codex honesty / wording critique",
    117: "AI emotion simulation / continuity",
    118: "Math / regularization / function-space discussion",
    120: "Support ticket / session-limit complaint",
    129: "Claude monthly plan vs API credits",
    121: "Humanoid robots / Unitree / Boston Dynamics",
    122: "Pentagon / autonomous weapons / military AI",
    123: "Claude Code guardrails / behavior restrictions",
    125: "Perplexity / Samsung / misc status",
    126: "MCP roadmap / AI-career planning",
    127: "Pro upgrade / worth-it discussion",
    85: "AI discomfort / artificiality / Guardian discussion",

    75: "ChatGPT image / photorealistic scene post",
    83: "Using AI for aerospace lecture notes",
    82: "Model changes / pricing / transparency",
    38: "Apple / Gmail / Google Sheets workflow",
    37: "MLX / Mac inference performance",
    36: "LocalLLaMA scraped comment noise",
    34: "Anthropic pricing / customer consumption",
    33: "Music / Suno / media-gen discussion",
    32: "Human slop / black-box criticism",
    31: "Weekly limits / infrastructure / paid tiers",
    30: "OCR / screenshots / prompt workflow",
    29: "Vibe-coded app launch",
    26: "Gemma vs Qwen / small-model comparison",
    25: "MoE / V100 / GPU offloading",
    24: "Prompting / user behavior / workflow advice",
    23: "Skills / tools / agent usage",
    22: "Multi-agent coordination / orchestrators",
    19: "AI / faith / philosophy discussion",
    18: "Agentic AI / autopilot / shipping",
    16: "vLLM / llama / image-preview noise",
    15: "iOS app planning / pro-limit issue",
    14: "Coding agent features / product ideas",
    13: "Biological learning / alignment / safety",
    12: "MLX / Apple Silicon / Gemma",
    10: "LLMs vs calculators / tools debate",
    9: "Qwen / Gemma / RAM requirements",
    8: "Qwen coder / GGUF / cline / Claude API",
    6: "Codex API / 500 errors / quota burn",
    5: "Cowork / agents / multi-machine experiment",
    4: "Claude Enterprise / access controls / compliance",
    3: "Formatting / punctuation / misc usage",
    2: "Wikipedia retrieval / factuality issue",
    39: "Playbook / world-data image noise",
    41: "ChatGPT image prompts / AI autonomy",
    43: "General multilingual AI discussion",
    44: "Apple Reminders / native workflow",
    80: "Token caching / token consumption complaints",
    79: "Opus UI/UX / app prototype",
    77: "Claude Code leak / copyright / fair use",
    76: "Using LLMs for document review",
    1: "General chatbot / AI discussion",
    74: "Smarter AI / existential dread",
    73: "Claude chat rules / summaries / roleplay files",
    72: "Evolution / biology / alignment",
    70: "Subagents / thinking blocks / JSONL logs",
    67: "Keeping up with new papers",
    66: "Barrier to entry / tool costs",
    65: "Tumor / CT / medical-imaging metrics",
    64: "Metric learning / cosine / classifiers",
    63: "Extended thinking / token limits",
    62: "Computer Use launch / capability discussion",
    61: "Benchmark / IQ-test / AGI debate",
    60: "Serverless / DB / email / workflow tooling",
    57: "Vertex AI / Gemini Flash / Codex API",
    56: "Gemini / image-preview noise",
    54: "SVG / Gemini / UI-gen / image noise",
    53: "Sora / video generation / storyboard",
    52: "Meta / visualization / response-model research",
    51: "Prompting / copywriting lessons",
    50: "Google Maps / apartment / image noise",
    49: "Python / C++ tracking benchmarks",
    48: "Metal / GPU / MLX / low-level ML",
    47: "Robotics dataset / warehouse / checkpoint",
    46: "Computer Use / upload file / job application",
    149: "Claude coding limitations / bug complaints",

    128: "Empty cluster / unused",
}
# load data
posts = pd.read_csv(POSTS_PATH)
comments = pd.read_csv(COMMENTS_PATH)

# make sure cluster_id is numeric where possible
posts["cluster_id"] = pd.to_numeric(posts["cluster_id"], errors="coerce").astype("Int64")

# assign topic labels to posts
posts["topic_label"] = posts["cluster_id"].map(CLUSTER_LABELS)

# build lookup table from posts
post_lookup = posts[["permalink", "cluster_id", "topic_label"]].drop_duplicates()

# merge labels into comments using permalink match
comments = comments.merge(
    post_lookup,
    left_on="post_permalink",
    right_on="permalink",
    how="left",
    suffixes=("", "_from_post")
)

# optional: remove duplicate permalink column after merge
comments = comments.drop(columns=["permalink"], errors="ignore")

# save outputs
posts.to_csv(POSTS_OUT, index=False)
comments.to_csv(COMMENTS_OUT, index=False)

print("Saved:", POSTS_OUT)
print("Saved:", COMMENTS_OUT)

print("\nPosts label counts:")
print(posts["topic_label"].value_counts(dropna=False))

print("\nComments label counts:")
print(comments["topic_label"].value_counts(dropna=False))