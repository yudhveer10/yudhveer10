# GitHub Profile Terminal Project

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
