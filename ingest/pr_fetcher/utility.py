import re
import html

GHSTACK_RE      = re.compile(r"^Stack from \[ghstack\]\([^)]+\).*(?:\n|$)", re.IGNORECASE | re.DOTALL)
MENTION_RE      = re.compile(r"(^|\s)@\w[\w-]+")
MD_LINK_RE      = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")         # [text](url) -> text
MD_IMAGE_RE     = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")        # ![alt](src) -> alt
CODEBLOCK_RE    = re.compile(r"```.*?```", re.DOTALL)            # fenced code blocks
INLINE_CODE_RE  = re.compile(r"`([^`]+)`")                       # `inline code` -> inline code
HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
QUOTE_LINE_RE   = re.compile(r"^\s*>\s.*$", re.MULTILINE)        # remove quoted lines
EMOJI_RE        = re.compile(r"[\U00010000-\U0010ffff]", re.UNICODE)  # strip astral emojis
MD_HEADING_RE = re.compile(r"^#{1,6}\s*", flags=re.MULTILINE)
MD_BOLD_ITALIC_RE = re.compile(r"(\*\*|\*|__|_)(.*?)\1")
def clean_body(text: str) -> str:
    if not text:
        return ""

    # Normalize escapes & HTML entities
    text = text.replace("\\u003E", ">")
    text = html.unescape(text)

    # Remove boilerplate / noise
    text = HTML_COMMENT_RE.sub("", text)
    text = GHSTACK_RE.sub("", text)
    text = CODEBLOCK_RE.sub("", text)
    text = QUOTE_LINE_RE.sub("", text)

    # Markdown simplifications
    text = MD_IMAGE_RE.sub(r"\1", text)   # keep alt text
    text = MD_LINK_RE.sub(r"\1", text)    # keep link text
    text = INLINE_CODE_RE.sub(r"\1", text)

    # Mentions & bullet noise like "* -> #123"
    text = MENTION_RE.sub(" ", text)
    text = re.sub(r"^\s*[\-\*]\s*(?:#\d+|->|—|–).*$", "", text, flags=re.MULTILINE)

    # Strip emojis (optional)
    text = EMOJI_RE.sub("", text)

    # strip markdown headings and bold/italic markers
    text = MD_HEADING_RE.sub("", text)
    text = MD_BOLD_ITALIC_RE.sub(r"\2", text)

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_meaningful_desc(text: str) -> bool:
    # e.g., at least 30 chars and 5 words after cleaning
    return len(text) >= 30 and len(text.split()) >= 5


def pr_is_complete(pr_obj):
    has_author = bool((pr_obj.get("user") or {}).get("login"))
    has_title = bool(pr_obj.get("title"))
    raw_body = pr_obj.get("body") or ""
    # has_body = is_meaningful_desc(body_clean)
    has_labels = len(pr_obj.get("labels", [])) > 0
    # has_reviewers = len(reviewers) > 0
    was_merged = pr_obj.get("merged_at") is not None  # your simplified merge check
    print(all([has_author, has_title, raw_body, has_labels, was_merged]))
    return all([has_author, has_title, raw_body, has_labels, was_merged])

