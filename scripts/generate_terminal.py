from __future__ import annotations

import html
import json
import os
import statistics
import textwrap
import urllib.error
import urllib.request
from base64 import b64encode
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageOps


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
IMAGE_PATH = ASSETS / "YudhveerP.jpg"
AVATAR_SOURCE_PATH = ASSETS / "avatar-source.png"
ASCII_PATH = ASSETS / "ascii.txt"
SVG_PATH = ASSETS / "terminal.svg"
DASHBOARD_PATH = ASSETS / "github-dashboard.svg"
PORTRAIT_PATH = ASSETS / "animated-avatar.png"

USERNAME = "yudhveer10"

ASCII_WIDTH = 48
ASCII_RAMP = " .:-=+*#%@"
SVG_WIDTH = 1100
SVG_HEIGHT = 500


@dataclass(frozen=True)
class Profile:
    name: str = "Yudhveer Singh Panwar"
    username: str = USERNAME
    role: str = "AI Engineer"
    company: str = "TechAivv Technologies"
    location: str = "Delhi, India"
    email: str = "yudhveerp10@gmail.com"
    linkedin: str = "linkedin.com/in/yudhveer-singh-panwar-504339265"
    leetcode: str = "leetcode.com/u/yudhveerpanwar"
    languages: tuple[str, ...] = ("Python", "JavaScript", "TypeScript", "SQL", "C++", "C")
    frontend: tuple[str, ...] = ("Next.js", "React.js", "Tailwind CSS")
    backend: tuple[str, ...] = ("Python APIs", "Express.js", "Real-time AI Inference", "Automation Pipelines")
    databases: tuple[str, ...] = ("SQL", "NoSQL", "Redis", "Firebase", "MongoDB")
    ai_stack: tuple[str, ...] = (
        "Agentic AI",
        "Generative Workflows",
        "LLM Orchestration",
        "Google Gemini 2.5",
        "TensorFlow/Keras",
        "CNNs",
        "OCR",
    )
    devops: tuple[str, ...] = ("AWS", "Docker", "Kubernetes", "CI/CD")
    projects: tuple[str, ...] = (
        "Flo.AI",
        "The Crop Doctor",
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


def crop_to_ascii_portrait(image: Image.Image) -> Image.Image:
    """Use a tighter head-and-shoulders crop so the ASCII portrait is readable."""
    alpha = image.getchannel("A")
    bbox = alpha.point(lambda value: 255 if value > 16 else 0).getbbox()
    if not bbox:
        return image

    left, top, right, bottom = bbox
    subject_width = right - left
    subject_height = bottom - top
    center_x = (left + right) // 2

    crop_width = int(subject_width * 0.58)
    crop_height = int(subject_height * 0.46)
    crop_left = center_x - crop_width // 2
    crop_right = center_x + crop_width // 2
    crop_bottom = top + crop_height

    box = (
        max(0, crop_left),
        max(0, top - int(subject_height * 0.015)),
        min(image.size[0], crop_right),
        min(image.size[1], crop_bottom),
    )
    return image.crop(box)


def make_terminal_portrait() -> Image.Image:
    """Draw a fictional illustrated avatar instead of reusing the real photo."""
    scale = 3
    canvas = Image.new("RGBA", (340 * scale, 320 * scale), (7, 16, 12, 255))

    def box(values: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        return tuple(value * scale for value in values)

    def pts(values: list[tuple[int, int]]) -> list[tuple[int, int]]:
        return [(x * scale, y * scale) for x, y in values]

    # Terminal panel glow and subtle scanlines.
    for radius, alpha in ((120, 26), (72, 34), (34, 42)):
        glow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        glow_draw = ImageDraw.Draw(glow, "RGBA")
        glow_draw.ellipse(box((62, 34, 278, 292)), fill=(0, 255, 136, alpha))
        glow = glow.filter(ImageFilter.GaussianBlur(radius * scale // 12))
        canvas = Image.alpha_composite(canvas, glow)

    draw = ImageDraw.Draw(canvas, "RGBA")

    for y in range(20, 304, 16):
        draw.line((28 * scale, y * scale, 312 * scale, y * scale), fill=(0, 255, 136, 12), width=1 * scale)
    draw.rounded_rectangle(box((26, 24, 314, 296)), radius=18 * scale, outline=(0, 255, 136, 70), width=2 * scale)

    # Body and shoulders: original animated AI engineer, not a photo trace.
    draw.polygon(
        pts([(78, 284), (108, 192), (136, 168), (204, 168), (232, 192), (262, 284)]),
        fill=(15, 23, 42, 255),
        outline=(41, 63, 83, 255),
    )
    draw.polygon(pts([(119, 185), (170, 260), (221, 185), (204, 174), (170, 214), (136, 174)]), fill=(7, 16, 12, 255))
    draw.line(pts([(170, 214), (170, 286)]), fill=(0, 255, 136, 80), width=2 * scale)
    draw.rounded_rectangle(box((92, 240, 248, 286)), radius=20 * scale, fill=(10, 18, 31, 255), outline=(0, 255, 136, 70), width=2 * scale)
    draw.rectangle(box((118, 254, 222, 262)), fill=(0, 255, 136, 24))
    draw.rectangle(box((142, 266, 198, 272)), fill=(88, 166, 255, 36))

    # Neck and face.
    skin = (198, 132, 95, 255)
    skin_light = (229, 165, 122, 255)
    shadow = (101, 53, 44, 110)
    draw.rounded_rectangle(box((150, 151, 190, 190)), radius=12 * scale, fill=(166, 98, 76, 255))
    draw.ellipse(box((102, 58, 238, 194)), fill=skin, outline=(255, 234, 206, 180), width=2 * scale)
    draw.ellipse(box((122, 78, 220, 168)), fill=skin_light)
    draw.pieslice(box((102, 112, 238, 212)), 0, 180, fill=shadow)

    # Hair, eyebrows, eyes, beard, and smile in cartoon proportions.
    hair = (18, 24, 32, 255)
    hair_hi = (55, 65, 78, 220)
    draw.pieslice(box((98, 38, 242, 134)), 178, 360, fill=hair)
    draw.polygon(pts([(106, 90), (124, 48), (154, 40), (196, 42), (230, 68), (238, 104), (218, 92), (198, 76), (162, 70), (130, 82)]), fill=hair)
    draw.arc(box((132, 50, 216, 100)), 190, 340, fill=hair_hi, width=3 * scale)
    draw.line(pts([(126, 116), (152, 110)]), fill=(21, 27, 36, 255), width=4 * scale)
    draw.line(pts([(188, 110), (214, 116)]), fill=(21, 27, 36, 255), width=4 * scale)
    draw.ellipse(box((137, 124, 149, 136)), fill=(8, 13, 18, 255))
    draw.ellipse(box((191, 124, 203, 136)), fill=(8, 13, 18, 255))
    draw.ellipse(box((141, 126, 145, 130)), fill=(255, 255, 255, 210))
    draw.ellipse(box((195, 126, 199, 130)), fill=(255, 255, 255, 210))
    draw.line(pts([(170, 130), (164, 150), (177, 150)]), fill=(122, 68, 55, 180), width=2 * scale)
    draw.pieslice(box((122, 130, 218, 202)), 0, 180, fill=(31, 31, 35, 210))
    draw.polygon(pts([(124, 162), (146, 194), (194, 194), (216, 162), (198, 184), (142, 184)]), fill=(30, 30, 34, 230))
    draw.arc(box((146, 150, 194, 176)), 18, 162, fill=(246, 238, 220, 230), width=3 * scale)

    # Glasses and small circuit accents make it character-like and techy.
    draw.rounded_rectangle(box((128, 116, 158, 143)), radius=9 * scale, outline=(0, 255, 136, 140), width=2 * scale)
    draw.rounded_rectangle(box((182, 116, 212, 143)), radius=9 * scale, outline=(0, 255, 136, 140), width=2 * scale)
    draw.line(pts([(158, 129), (182, 129)]), fill=(0, 255, 136, 120), width=2 * scale)
    draw.line(pts([(230, 112), (258, 102), (276, 102)]), fill=(0, 255, 136, 120), width=2 * scale)
    draw.ellipse(box((274, 98, 282, 106)), fill=(0, 255, 136, 220))
    draw.line(pts([(110, 208), (78, 208), (60, 224)]), fill=(0, 255, 136, 100), width=2 * scale)
    draw.ellipse(box((54, 218, 66, 230)), fill=(0, 255, 136, 190))

    # Crisp cartoon outline.
    outline = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    outline_draw = ImageDraw.Draw(outline, "RGBA")
    outline_draw.ellipse(box((97, 54, 243, 198)), outline=(0, 255, 136, 120), width=3 * scale)
    outline_draw.polygon(pts([(78, 284), (108, 192), (136, 168), (204, 168), (232, 192), (262, 284)]), outline=(0, 255, 136, 120))
    canvas = Image.alpha_composite(canvas, outline.filter(ImageFilter.GaussianBlur(0.25 * scale)))

    vignette = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    vignette_alpha = Image.new("L", canvas.size, 0)
    ImageDraw.Draw(vignette_alpha).rectangle(box((0, 0, 340, 24)), fill=90)
    ImageDraw.Draw(vignette_alpha).rectangle(box((0, 288, 340, 320)), fill=85)
    vignette_alpha = vignette_alpha.filter(ImageFilter.GaussianBlur(16 * scale))
    vignette.putalpha(vignette_alpha)
    canvas = Image.alpha_composite(canvas, vignette)

    return canvas.resize((340, 320), Image.Resampling.LANCZOS)


def image_to_data_uri(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    encoded = b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def image_to_png_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def enhance_for_ascii(image: Image.Image) -> tuple[Image.Image, Image.Image]:
    subject = crop_to_ascii_portrait(crop_to_subject(image))
    alpha = subject.getchannel("A")

    gray = subject.convert("L")
    gray = ImageOps.equalize(gray)
    gray = ImageOps.autocontrast(gray, cutoff=1)
    gray = ImageEnhance.Contrast(gray).enhance(1.55)
    gray = ImageEnhance.Sharpness(gray).enhance(2.0)
    gray = gray.filter(ImageFilter.MedianFilter(3))

    edges = gray.filter(ImageFilter.FIND_EDGES)
    edges = ImageOps.autocontrast(edges)
    edge_mask = ImageOps.invert(edges).point(lambda value: max(value, 86))
    gray = ImageChops.multiply(gray, edge_mask)
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
        "repos": "21",
        "stars": "1",
        "followers": "4",
        "following": "7",
        "top_language": "HTML",
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


def pill(label: str, x: int, y: int, width: int) -> str:
    return f'''
      <rect x="{x}" y="{y}" width="{width}" height="34" rx="8" fill="#111a20" stroke="#23313b"/>
      <text x="{x + 16}" y="{y + 22}" class="mono tag">{esc(label)}</text>
    '''


def stat_value(value: str) -> str:
    return "--" if value == "dynamic" else value


def build_svg(ascii_art: str, profile: Profile, stats: dict[str, str], portrait_uri: str) -> str:
    ascii_panel_x = 46
    ascii_panel_y = 96
    ascii_panel_height = 344
    ascii_line_height = 10
    ascii_lines = ascii_art.splitlines()[:32]
    ascii_start_y = int(ascii_panel_y + (ascii_panel_height - ((len(ascii_lines) - 1) * ascii_line_height)) / 2)
    ascii_spans = []
    for index, line in enumerate(ascii_lines):
        y = ascii_start_y + index * ascii_line_height
        ascii_spans.append(f'<tspan x="68" y="{y}">{esc(line)}</tspan>')

    stack_lines = [
        "agentic ai  llm orchestration  gemini 2.5",
        "next.js  react  python  express.js",
        "tensorflow/keras  cnns  ocr  docker",
    ]

    project_lines = [
        "Flo.AI",
        "The Crop Doctor",
        "AI Agents",
        "Automation Pipelines",
    ]

    top_language = stats["top_language"] if stats["top_language"] != "AI / Full Stack" else "AI"

    return f'''<svg width="{SVG_WIDTH}" height="{SVG_HEIGHT}" viewBox="0 0 {SVG_WIDTH} {SVG_HEIGHT}" fill="none" xmlns="http://www.w3.org/2000/svg" role="img" aria-labelledby="title desc">
  <title id="title">{esc(profile.name)} terminal profile</title>
  <desc id="desc">Generated terminal-style GitHub profile with an illustrated character avatar and system information.</desc>
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
      .title {{ fill: #f0f6fc; font-size: 31px; font-weight: 800; }}
      .subtitle {{ fill: #8b949e; font-size: 17px; }}
      .body {{ fill: #d7fbe8; font-size: 15px; }}
      .muted {{ fill: #7d8590; }}
      .cmd {{ fill: #00ff88; font-size: 17px; font-weight: 800; }}
      .tag {{ fill: #d7fbe8; font-size: 14px; font-weight: 700; }}
      .ascii {{ fill: #b7ffd0; font-size: 9px; letter-spacing: 0; filter: url(#softGlow); }}
      .prompt {{ fill: #00ff88; font-size: 15px; font-weight: 800; }}
      .cursor {{ fill: #00ff88; }}
    </style>
  </defs>

  <rect x="14" y="14" width="1072" height="472" rx="24" fill="#030507"/>
  <rect x="24" y="24" width="1052" height="452" rx="18" fill="url(#terminalBg)" stroke="#263241" stroke-width="2"/>
  <rect x="24" y="24" width="1052" height="52" rx="18" fill="url(#titleBar)"/>
  <path d="M24 58H1076V76H24V58Z" fill="#111820"/>

  <circle cx="56" cy="51" r="8" fill="#ff5f57"/>
  <circle cx="82" cy="51" r="8" fill="#ffbd2e"/>
  <circle cx="108" cy="51" r="8" fill="#28c840"/>
  <text x="550" y="57" text-anchor="middle" class="mono muted" font-size="14">yudhveer10 / README.md</text>

  <clipPath id="portraitClip">
    <rect x="58" y="108" width="336" height="320" rx="10"/>
  </clipPath>
  <rect x="{ascii_panel_x}" y="{ascii_panel_y}" width="360" height="{ascii_panel_height}" rx="12" fill="#07100c" stroke="#1d2b24"/>
  <image x="58" y="108" width="336" height="320" href="{portrait_uri}" clip-path="url(#portraitClip)" preserveAspectRatio="xMidYMid meet"/>
  <rect x="58" y="108" width="336" height="320" rx="10" fill="none" stroke="#00ff88" stroke-opacity="0.25"/>
  <text x="70" y="416" class="mono muted" font-size="12">illustrated AI avatar generated by script</text>

  <text x="448" y="126" class="mono cmd">$ whoami</text>
  <text x="448" y="166" class="mono title">{esc(profile.name)}</text>
  <text x="450" y="195" class="mono subtitle">{esc(profile.role)} at {esc(profile.company)} | {esc(profile.location)}</text>
  <text x="450" y="228" class="mono body">contact : {esc(profile.email)}</text>

  <text x="448" y="330" class="mono cmd">$ stack</text>
  {tspan_lines(stack_lines, 450, 364, 22)}

  <text x="754" y="330" class="mono cmd">$ projects</text>
  {tspan_lines(project_lines, 756, 364, 22)}

  {pill(f"repos {stat_value(stats['repos'])}", 448, 258, 104)}
  {pill(f"stars {stat_value(stats['stars'])}", 564, 258, 108)}
  {pill(f"followers {stat_value(stats['followers'])}", 684, 258, 136)}
  {pill(f"top {top_language}", 832, 258, 104)}

  <text x="448" y="450" class="mono prompt">yudhveer@github</text>
  <text x="578" y="450" class="mono body">:</text>
  <text x="590" y="450" class="mono" fill="#58a6ff" font-size="15">~/ai-lab</text>
  <text x="666" y="450" class="mono prompt">$</text>
  <text x="688" y="450" class="mono body">building agentic AI systems</text>
  <rect x="928" y="436" width="10" height="20" class="cursor">
    <animate attributeName="opacity" values="1;1;0;0;1" dur="1.2s" repeatCount="indefinite"/>
  </rect>
</svg>
'''


def build_dashboard_svg(profile: Profile, stats: dict[str, str]) -> str:
    repos = stat_value(stats["repos"])
    stars = stat_value(stats["stars"])
    followers = stat_value(stats["followers"])
    following = stat_value(stats["following"])
    top_language = stats["top_language"] if stats["top_language"] != "AI / Full Stack" else "AI"

    return f'''<svg width="1100" height="280" viewBox="0 0 1100 280" fill="none" xmlns="http://www.w3.org/2000/svg" role="img" aria-labelledby="dashboard-title dashboard-desc">
  <title id="dashboard-title">{esc(profile.name)} GitHub dashboard</title>
  <desc id="dashboard-desc">Generated GitHub analytics dashboard card.</desc>
  <defs>
    <linearGradient id="dashBg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#07100c"/>
      <stop offset="48%" stop-color="#0d1117"/>
      <stop offset="100%" stop-color="#111827"/>
    </linearGradient>
    <filter id="dashGlow" x="-20%" y="-20%" width="140%" height="140%">
      <feDropShadow dx="0" dy="0" stdDeviation="8" flood-color="#00ff88" flood-opacity="0.18"/>
    </filter>
    <style>
      .mono {{ font-family: "SFMono-Regular", "Cascadia Code", "Consolas", "Liberation Mono", monospace; }}
      .title {{ fill: #f0f6fc; font-size: 28px; font-weight: 800; }}
      .label {{ fill: #8b949e; font-size: 14px; font-weight: 700; }}
      .value {{ fill: #00ff88; font-size: 34px; font-weight: 900; }}
      .small {{ fill: #d7fbe8; font-size: 15px; }}
      .cmd {{ fill: #00ff88; font-size: 16px; font-weight: 800; }}
    </style>
  </defs>

  <rect x="12" y="12" width="1076" height="256" rx="22" fill="#030507"/>
  <rect x="24" y="24" width="1052" height="232" rx="18" fill="url(#dashBg)" stroke="#263241" stroke-width="2"/>
  <text x="54" y="70" class="mono cmd">$ github_summary --local</text>
  <text x="54" y="112" class="mono title">@{esc(profile.username)}</text>
  <text x="54" y="144" class="mono small">{esc(profile.role)} | {esc(profile.location)} | Agentic AI and scalable SaaS</text>
  <text x="54" y="198" class="mono small">Resume-backed stack: LLM orchestration, real-time inference, OCR automation.</text>

  <g filter="url(#dashGlow)">
    <rect x="430" y="58" width="140" height="132" rx="14" fill="#0d1713" stroke="#1d3a2c"/>
    <text x="500" y="96" text-anchor="middle" class="mono label">Repos</text>
    <text x="500" y="146" text-anchor="middle" class="mono value">{esc(repos)}</text>
  </g>
  <g filter="url(#dashGlow)">
    <rect x="590" y="58" width="140" height="132" rx="14" fill="#0d1713" stroke="#1d3a2c"/>
    <text x="660" y="96" text-anchor="middle" class="mono label">Stars</text>
    <text x="660" y="146" text-anchor="middle" class="mono value">{esc(stars)}</text>
  </g>
  <g filter="url(#dashGlow)">
    <rect x="750" y="58" width="140" height="132" rx="14" fill="#0d1713" stroke="#1d3a2c"/>
    <text x="820" y="96" text-anchor="middle" class="mono label">Followers</text>
    <text x="820" y="146" text-anchor="middle" class="mono value">{esc(followers)}</text>
  </g>
  <g filter="url(#dashGlow)">
    <rect x="910" y="58" width="116" height="132" rx="14" fill="#0d1713" stroke="#1d3a2c"/>
    <text x="968" y="96" text-anchor="middle" class="mono label">Following</text>
    <text x="968" y="146" text-anchor="middle" class="mono value">{esc(following)}</text>
  </g>

  <rect x="430" y="208" width="596" height="28" rx="8" fill="#101820" stroke="#23313b"/>
  <text x="450" y="227" class="mono small">top language: {esc(top_language)}   focus: agents, automation, AI SaaS</text>
</svg>
'''


def write_readme(profile: Profile) -> None:
    content = f'''<div align="center">

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&height=260&color=0:020617,50:0d1117,100:00ff88&text=Yudhveer%20Singh%20Panwar&fontSize=48&fontColor=ffffff&animation=fadeIn&fontAlignY=38&desc=AI%20Engineer%20%7C%20Full%20Stack%20Builder%20%7C%20Agentic%20AI&descAlignY=58&descSize=18" alt="Yudhveer Singh Panwar">

[![Typing SVG](https://readme-typing-svg.demolab.com?font=Fira+Code&weight=700&size=24&duration=2400&pause=700&color=00FF88&center=true&vCenter=true&width=900&lines=Full+Stack+AI+Developer;Agentic+AI+%7C+Generative+Workflows;Next.js+%7C+Python+%7C+Gemini+2.5;Real-time+AI+Inference+and+Automation)](https://git.io/typing-svg)

</div>

<br>

<div align="center">
  <img src="./assets/terminal.svg" width="100%" alt="Generated terminal profile">
</div>

<br>

## About Me

```bash
> whoami

Name      : Yudhveer Singh Panwar
Username  : yudhveer10
Role      : AI Engineer
Company   : {profile.company}
Location  : Delhi, India
Focus     : Agentic AI, LLM orchestration, real-time inference, AI SaaS
Email     : {profile.email}
```

I build AI-powered products across the full stack, connecting machine learning systems with fast, user-facing applications. My work spans agentic workflow generation, scalable AI SaaS platforms, backend automation, OCR pipelines, and deep learning diagnostics.

## Tech Arsenal

<div align="center">

### Languages
<img src="https://skillicons.dev/icons?i=python,cpp,js,ts,mysql" alt="Languages">

### Frontend
<img src="https://skillicons.dev/icons?i=react,nextjs,tailwind,html,css" alt="Frontend">

### Backend
<img src="https://skillicons.dev/icons?i=nodejs,express,python" alt="Backend">

### Databases
<img src="https://skillicons.dev/icons?i=mongodb,mysql,redis,firebase" alt="Databases">

### AI / ML / Automation
<img src="https://img.shields.io/badge/Agentic_AI-111827?style=for-the-badge&logoColor=white" alt="Agentic AI">
<img src="https://img.shields.io/badge/LLM_Orchestration-00ff88?style=for-the-badge&logoColor=black" alt="LLM Orchestration">
<img src="https://img.shields.io/badge/Generative_Workflows-111827?style=for-the-badge&logoColor=white" alt="Generative Workflows">
<img src="https://img.shields.io/badge/Gemini-4285F4?style=for-the-badge&logo=googlegemini&logoColor=white" alt="Gemini">
<img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow">
<img src="https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white" alt="Keras">
<img src="https://img.shields.io/badge/CNNs-111827?style=for-the-badge&logoColor=white" alt="CNNs">
<img src="https://img.shields.io/badge/OCR_Automation-00ff88?style=for-the-badge&logoColor=black" alt="OCR Automation">

### Cloud / DevOps
<img src="https://skillicons.dev/icons?i=aws,docker,kubernetes" alt="Cloud and DevOps">
<img src="https://img.shields.io/badge/CI%2FCD-111827?style=for-the-badge&logo=githubactions&logoColor=white" alt="CI/CD">

</div>

## Current Builds

| Project | What it is |
| --- | --- |
| Flo.AI | Agentic AI framework using Gemini 2.5 for 15+ step workflow automation |
| The Crop Doctor | CNN-based crop disease diagnostics trained on 5,000+ annotated images |
| AI Automation Pipelines | Python, Pandas, NumPy, OCR, scheduling, and API integration workflows |
| Real-time AI SaaS | Full-stack AI platform work with responsive UI and low-latency inference |

## GitHub Analytics

<div align="center">

<img src="./assets/github-dashboard.svg" width="100%" alt="Generated GitHub analytics dashboard">

<br><br>

<img src="https://streak-stats.demolab.com?user={profile.username}&theme=chartreuse-dark&hide_border=true&background=0d1117&ring=00ff88&fire=00ff88&currStreakLabel=00ff88" alt="GitHub streak">

</div>

## Contribution Graph

<div align="center">
  <img src="https://github-readme-activity-graph.vercel.app/graph?username={profile.username}&theme=github-compact&hide_border=true&bg_color=0d1117&color=00ff88&line=00ff88&point=ffffff" width="100%" alt="Contribution graph">
</div>

## Contribution Snake

<div align="center">
  <img src="https://raw.githubusercontent.com/{profile.username}/{profile.username}/output/github-contribution-grid-snake-dark.svg" width="100%" alt="Contribution snake">
</div>

## Highlights

| Area | Signal |
| --- | --- |
| Agentic AI | Gemini 2.5, LLM orchestration, generative workflow automation |
| Full Stack AI | Next.js, React, Python, Express.js, real-time AI inference |
| Automation | Pandas, NumPy, OCR, backend scheduling, API integrations |
| Deep Learning | TensorFlow/Keras, CNNs, image classification, web integration |
| Cloud & Delivery | AWS, Docker, Kubernetes, CI/CD |

## Connect

<div align="center">

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/yudhveer-singh-panwar-504339265/)
[![Email](https://img.shields.io/badge/Email-111827?style=for-the-badge&logo=gmail&logoColor=EA4335)](mailto:{profile.email})
[![LeetCode](https://img.shields.io/badge/LeetCode-FFA116?style=for-the-badge&logo=leetcode&logoColor=black)](https://leetcode.com/u/yudhveerpanwar/)

</div>

<br>

<div align="center">
  <img width="100%" src="https://capsule-render.vercel.app/api?type=waving&height=120&section=footer&color=0:020617,50:0d1117,100:00ff88" alt="Footer">
</div>
'''
    (ROOT / "README.md").write_text(content, encoding="utf-8")


def write_project_context() -> None:
    content = f"""# GitHub Profile Terminal Project

This repository generates a terminal-style GitHub profile with an illustrated AI avatar and automated profile assets.

## Source Of Truth

Do not manually edit `assets/terminal.svg`, `assets/animated-avatar.png`, `assets/github-dashboard.svg`, or `assets/ascii.txt`.

Regenerate generated assets with:

```bash
python scripts/generate_terminal.py
```

## Pipeline

1. Read `assets/YudhveerP.jpg` for the generated ASCII text asset
2. Remove plain background, or use `rembg` automatically if installed
3. Crop subject
4. Apply histogram equalization, contrast enhancement, sharpening, noise reduction, and edge emphasis
5. Generate `assets/ascii.txt`
6. Draw `assets/animated-avatar.png` as a fictional illustrated AI character
7. Build `assets/terminal.svg`
8. Update `README.md` with the full premium profile layout and generated terminal centerpiece

## Profile

- Name: Yudhveer Singh Panwar
- Username: yudhveer10
- Role: AI Engineer
- Company: TechAivv Technologies
- Location: Delhi, India
- Email: yudhveerp10@gmail.com
- LinkedIn: https://linkedin.com/in/yudhveer-singh-panwar-504339265/
- LeetCode: https://leetcode.com/u/yudhveerpanwar/

## Generated Files

- `assets/ascii.txt`
- `assets/terminal.svg`
- `assets/animated-avatar.png`
- `assets/github-dashboard.svg`
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
    portrait = make_terminal_portrait()
    PORTRAIT_PATH.write_bytes(image_to_png_bytes(portrait))

    stats = fetch_github_stats(profile.username)
    SVG_PATH.write_text(build_svg(ascii_art, profile, stats, image_to_data_uri(portrait)), encoding="utf-8")
    DASHBOARD_PATH.write_text(build_dashboard_svg(profile, stats), encoding="utf-8")
    write_readme(profile)
    write_project_context()

    print(f"Generated {ASCII_PATH.relative_to(ROOT)}")
    print(f"Generated {PORTRAIT_PATH.relative_to(ROOT)}")
    print(f"Generated {SVG_PATH.relative_to(ROOT)}")
    print(f"Generated {DASHBOARD_PATH.relative_to(ROOT)}")
    print("Updated README.md")


if __name__ == "__main__":
    main()
