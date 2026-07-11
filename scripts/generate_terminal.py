from __future__ import annotations

import html
import json
import os
import statistics
import textwrap
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageEnhance, ImageFilter, ImageOps


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
IMAGE_PATH = ASSETS / "YudhveerP.jpg"
ASCII_PATH = ASSETS / "ascii.txt"
SVG_PATH = ASSETS / "terminal.svg"

USERNAME = "yudhveer10"

ASCII_WIDTH = 58
ASCII_RAMP = " .:-=+*#%@"
SVG_WIDTH = 1280
SVG_HEIGHT = 720


@dataclass(frozen=True)
class Profile:
    name: str = "Yudhveer Singh Panwar"
    username: str = USERNAME
    role: str = "AI Engineer"
    company: str = "TechAIVV"
    location: str = "Delhi, India"
    email: str = "yudhveerp10@gmail.com"
    linkedin: str = "linkedin.com/in/yudhveer-singh-panwar-504339265"
    leetcode: str = "leetcode.com/u/yudhveerpanwar"
    languages: tuple[str, ...] = ("Python", "C++", "JavaScript", "TypeScript", "SQL")
    frontend: tuple[str, ...] = ("React", "Next.js", "Tailwind CSS")
    backend: tuple[str, ...] = ("FastAPI", "Node.js", "Express")
    databases: tuple[str, ...] = ("MongoDB", "PostgreSQL", "MySQL")
    ai_stack: tuple[str, ...] = ("LangGraph", "Gemini", "OpenAI", "TensorFlow", "FAISS", "Scikit-Learn")
    projects: tuple[str, ...] = (
        "HireAIVV",
        "AI Content Repurposing Platform",
        "LangGraph Workflows",
        "AI Agents",
        "Panwar Alpha",
    )


def load_photo(path: Path) -> Image.Image:
    if not path.exists():
        raise FileNotFoundError(f"Missing profile photo: {path}")
    image = Image.open(path).convert("RGBA")
    return remove_background(image)


def remove_background(image: Image.Image) -> Image.Image:
    """Remove plain backgrounds locally, and use rembg automatically when available."""
    try:
        from rembg import remove  # type: ignore

        return remove(image).convert("RGBA")
    except Exception:
        pass

    rgb = image.convert("RGB")
    arr = np.asarray(rgb).astype(np.int16)

    light = (arr[:, :, 0] > 226) & (arr[:, :, 1] > 226) & (arr[:, :, 2] > 226)
    low_chroma = (arr.max(axis=2) - arr.min(axis=2)) < 32
    bg = light & low_chroma

    alpha = np.where(bg, 0, 255).astype(np.uint8)
    alpha_img = Image.fromarray(alpha, "L")
    alpha_img = alpha_img.filter(ImageFilter.MedianFilter(5))
    alpha_img = alpha_img.filter(ImageFilter.GaussianBlur(0.8))

    out = image.copy()
    out.putalpha(alpha_img)
    return out


def crop_to_subject(image: Image.Image) -> Image.Image:
    alpha = image.getchannel("A")
    bbox = alpha.point(lambda value: 255 if value > 16 else 0).getbbox()
    if not bbox:
        return image

    left, top, right, bottom = bbox
    width, height = image.size
    pad_x = int((right - left) * 0.06)
    pad_y = int((bottom - top) * 0.04)
    box = (
        max(0, left - pad_x),
        max(0, top - pad_y),
        min(width, right + pad_x),
        min(height, bottom + pad_y),
    )
    return image.crop(box)


def enhance_for_ascii(image: Image.Image) -> tuple[Image.Image, Image.Image]:
    subject = crop_to_subject(image)
    alpha = subject.getchannel("A")

    gray = subject.convert("L")
    gray = ImageOps.equalize(gray)
    gray = ImageOps.autocontrast(gray, cutoff=1)
    gray = ImageEnhance.Contrast(gray).enhance(1.45)
    gray = ImageEnhance.Sharpness(gray).enhance(1.8)
    gray = gray.filter(ImageFilter.MedianFilter(3))

    edges = gray.filter(ImageFilter.FIND_EDGES)
    edges = ImageOps.autocontrast(edges)
    gray = ImageChops.multiply(gray, ImageOps.invert(edges).point(lambda value: max(value, 64)))
    gray.putalpha(alpha)
    return gray.convert("RGBA"), alpha


def resize_for_ascii(image: Image.Image, width: int) -> Image.Image:
    w, h = image.size
    aspect = h / max(w, 1)
    height = max(1, int(width * aspect * 0.49))
    return image.resize((width, height), Image.Resampling.LANCZOS)


def image_to_ascii(image: Image.Image, width: int = ASCII_WIDTH) -> str:
    processed, _ = enhance_for_ascii(image)
    small = resize_for_ascii(processed, width)
    gray = np.asarray(small.convert("L"))
    alpha = np.asarray(small.getchannel("A"))

    chars: list[str] = []
    ramp_max = len(ASCII_RAMP) - 1
    for row in range(gray.shape[0]):
        line = []
        for col in range(gray.shape[1]):
            if alpha[row, col] < 40:
                line.append(" ")
                continue
            value = 255 - int(gray[row, col])
            index = min(ramp_max, max(0, round(value * ramp_max / 255)))
            line.append(ASCII_RAMP[index])
        chars.append("".join(line).rstrip())

    trimmed = trim_ascii(chars)
    return "\n".join(trimmed) + "\n"


def trim_ascii(lines: list[str]) -> list[str]:
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return lines


def fetch_github_stats(username: str) -> dict[str, str]:
    fallback = {
        "repos": "dynamic",
        "stars": "dynamic",
        "followers": "dynamic",
        "following": "dynamic",
        "top_language": "AI / Full Stack",
    }

    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "github-profile-terminal-generator",
    }
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    try:
        user = github_json(f"https://api.github.com/users/{username}", headers)
        repos = github_json(f"https://api.github.com/users/{username}/repos?per_page=100&type=owner", headers)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
        return fallback

    languages: list[str] = []
    stars = 0
    for repo in repos if isinstance(repos, list) else []:
        stars += int(repo.get("stargazers_count") or 0)
        language = repo.get("language")
        if language:
            languages.append(language)

    top_language = fallback["top_language"]
    if languages:
        top_language = statistics.mode(languages)

    return {
        "repos": str(user.get("public_repos", fallback["repos"])),
        "stars": str(stars),
        "followers": str(user.get("followers", fallback["followers"])),
        "following": str(user.get("following", fallback["following"])),
        "top_language": top_language,
    }


def github_json(url: str, headers: dict[str, str]) -> object:
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=8) as response:
        return json.loads(response.read().decode("utf-8"))


def esc(value: str) -> str:
    return html.escape(value, quote=True)


def tspan_lines(lines: list[str], x: int, y: int, line_height: int = 17) -> str:
    spans = []
    for index, line in enumerate(lines):
        dy = 0 if index == 0 else line_height
        spans.append(f'<tspan x="{x}" dy="{dy}">{esc(line)}</tspan>')
    return f'<text x="{x}" y="{y}" class="mono body">{"".join(spans)}</text>'


def section(title: str, lines: list[str], x: int, y: int) -> str:
    body = tspan_lines(lines, x, y + 28)
    return f'''
      <text x="{x}" y="{y}" class="mono cmd">$ {esc(title)}</text>
      {body}
    '''


def wrap_values(label: str, values: tuple[str, ...], width: int = 38) -> list[str]:
    joined = ", ".join(values)
    wrapped = textwrap.wrap(joined, width=width)
    if not wrapped:
        return [f"{label:<10}: -"]
    lines = [f"{label:<10}: {wrapped[0]}"]
    lines.extend(f"{'':<10}  {line}" for line in wrapped[1:])
    return lines


def wrapped_row(label: str, value: str, width: int = 46) -> list[str]:
    wrapped = textwrap.wrap(value, width=width) or ["-"]
    lines = [f"{label:<9}: {wrapped[0]}"]
    lines.extend(f"{'':<9}  {line}" for line in wrapped[1:])
    return lines


def build_svg(ascii_art: str, profile: Profile, stats: dict[str, str]) -> str:
    ascii_lines = ascii_art.splitlines()
    ascii_spans = []
    for index, line in enumerate(ascii_lines[:48]):
        y = 105 + index * 11
        ascii_spans.append(f'<tspan x="52" y="{y}">{esc(line)}</tspan>')

    whoami = []
    whoami.extend(wrapped_row("name", profile.name))
    whoami.extend(wrapped_row("role", profile.role))
    whoami.extend(wrapped_row("company", profile.company))
    whoami.extend(wrapped_row("location", profile.location))
    whoami.extend(wrapped_row("email", profile.email))
    whoami.extend(wrapped_row("linkedin", profile.linkedin))
    whoami.extend(wrapped_row("leetcode", profile.leetcode))

    stack = []
    stack.extend(wrap_values("languages", profile.languages))
    stack.extend(wrap_values("frontend", profile.frontend))
    stack.extend(wrap_values("backend", profile.backend))
    stack.extend(wrap_values("database", profile.databases))
    stack.extend(wrap_values("ai", profile.ai_stack))

    projects = [f"{index + 1}. {project}" for index, project in enumerate(profile.projects)]
    github = [
        f"username  : @{profile.username}",
        f"repos     : {stats['repos']}",
        f"stars     : {stats['stars']}",
        f"followers : {stats['followers']}",
        f"following : {stats['following']}",
        f"top lang  : {stats['top_language']}",
    ]

    return f'''<svg width="{SVG_WIDTH}" height="{SVG_HEIGHT}" viewBox="0 0 {SVG_WIDTH} {SVG_HEIGHT}" fill="none" xmlns="http://www.w3.org/2000/svg" role="img" aria-labelledby="title desc">
  <title id="title">{esc(profile.name)} terminal profile</title>
  <desc id="desc">Generated terminal-style GitHub profile with ASCII portrait and system information.</desc>
  <defs>
    <linearGradient id="terminalBg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#080c10"/>
      <stop offset="45%" stop-color="#0d1117"/>
      <stop offset="100%" stop-color="#101820"/>
    </linearGradient>
    <linearGradient id="titleBar" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%" stop-color="#151b23"/>
      <stop offset="100%" stop-color="#0f141b"/>
    </linearGradient>
    <filter id="softGlow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="0" stdDeviation="7" flood-color="#00ff88" flood-opacity="0.28"/>
    </filter>
    <style>
      .mono {{ font-family: "SFMono-Regular", "Cascadia Code", "Consolas", "Liberation Mono", monospace; }}
      .body {{ fill: #d7fbe8; font-size: 14.2px; }}
      .muted {{ fill: #7d8590; }}
      .cmd {{ fill: #00ff88; font-size: 16px; font-weight: 700; }}
      .ascii {{ fill: #c9ffe0; font-size: 10.2px; letter-spacing: 0; filter: url(#softGlow); }}
      .prompt {{ fill: #00ff88; font-size: 15px; font-weight: 700; }}
      .cursor {{ fill: #00ff88; }}
    </style>
  </defs>

  <rect x="16" y="16" width="1248" height="688" rx="24" fill="#030507"/>
  <rect x="26" y="26" width="1228" height="668" rx="18" fill="url(#terminalBg)" stroke="#263241" stroke-width="2"/>
  <rect x="26" y="26" width="1228" height="52" rx="18" fill="url(#titleBar)"/>
  <path d="M26 58H1254V78H26V58Z" fill="#111820"/>

  <circle cx="58" cy="52" r="8" fill="#ff5f57"/>
  <circle cx="84" cy="52" r="8" fill="#ffbd2e"/>
  <circle cx="110" cy="52" r="8" fill="#28c840"/>
  <text x="640" y="58" text-anchor="middle" class="mono muted" font-size="14">yudhveer10/profile-terminal - generated from assets/YudhveerP.jpg</text>

  <rect x="46" y="96" width="472" height="554" rx="12" fill="#07100c" stroke="#1d2b24"/>
  <text class="mono ascii">
    {''.join(ascii_spans)}
  </text>
  <text x="52" y="674" class="mono muted" font-size="13">ascii portrait: assets/ascii.txt</text>

  <line x1="552" y1="102" x2="552" y2="648" stroke="#243244"/>
  {section("whoami", whoami, 590, 118)}
  {section("tech_stack", stack, 590, 310)}
  {section("github_stats", github, 590, 498)}
  {section("projects", projects, 850, 498)}

  <text x="590" y="652" class="mono prompt">yudhveer@github</text>
  <text x="720" y="652" class="mono body">:</text>
  <text x="732" y="652" class="mono" fill="#58a6ff" font-size="15">~/profile</text>
  <text x="812" y="652" class="mono prompt">$</text>
  <text x="834" y="652" class="mono body">building AI systems</text>
  <rect x="1048" y="638" width="10" height="20" class="cursor">
    <animate attributeName="opacity" values="1;1;0;0;1" dur="1.2s" repeatCount="indefinite"/>
  </rect>
</svg>
'''


def write_readme() -> None:
    content = '''<div align="center">
  <img src="./assets/terminal.svg" width="100%" alt="Yudhveer Singh Panwar terminal profile">
</div>
'''
    (ROOT / "README.md").write_text(content, encoding="utf-8")


def write_project_context() -> None:
    content = f"""# GitHub Profile Terminal Project

This repository generates a terminal-style GitHub profile from `assets/YudhveerP.jpg`.

## Source Of Truth

Do not manually edit `assets/terminal.svg` or `assets/ascii.txt`.

Regenerate generated assets with:

```bash
python scripts/generate_terminal.py
```

## Pipeline

1. Read `assets/YudhveerP.jpg`
2. Remove plain background, or use `rembg` automatically if installed
3. Crop subject
4. Apply histogram equalization, contrast enhancement, sharpening, noise reduction, and edge emphasis
5. Generate `assets/ascii.txt`
6. Build `assets/terminal.svg`
7. Update `README.md` so it displays the generated terminal

## Profile

- Name: Yudhveer Singh Panwar
- Username: yudhveer10
- Role: AI Engineer
- Company: TechAIVV
- Location: Delhi, India
- Email: yudhveerp10@gmail.com
- LinkedIn: https://linkedin.com/in/yudhveer-singh-panwar-504339265/
- LeetCode: https://leetcode.com/u/yudhveerpanwar/

## Generated Files

- `assets/ascii.txt`
- `assets/terminal.svg`
- `README.md`

Generated on demand by `scripts/generate_terminal.py`.
"""
    (ROOT / "PROJECT_CONTEXT.md").write_text(content, encoding="utf-8")


def main() -> None:
    ASSETS.mkdir(exist_ok=True)
    profile = Profile()
    photo = load_photo(IMAGE_PATH)
    ascii_art = image_to_ascii(photo)
    ASCII_PATH.write_text(ascii_art, encoding="utf-8")

    stats = fetch_github_stats(profile.username)
    SVG_PATH.write_text(build_svg(ascii_art, profile, stats), encoding="utf-8")
    write_readme()
    write_project_context()

    print(f"Generated {ASCII_PATH.relative_to(ROOT)}")
    print(f"Generated {SVG_PATH.relative_to(ROOT)}")
    print("Updated README.md")


if __name__ == "__main__":
    main()
