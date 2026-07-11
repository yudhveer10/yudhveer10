# GitHub Profile Terminal Project

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
