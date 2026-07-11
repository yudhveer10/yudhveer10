from PIL import Image

IMAGE_PATH = "../assets/YudhveerP.jpg"
OUTPUT_FILE = "../assets/ascii.txt"

WIDTH = 90

ASCII = "@%#*+=-:. "


def resize(image, width):
    w, h = image.size

    ratio = h / w

    height = int(width * ratio * 0.55)

    return image.resize((width, height))


def gray(image):
    return image.convert("L")


def convert(image):
    pixels = image.getdata()

    chars = "".join(
        ASCII[pixel * (len(ASCII) - 1) // 255]
        for pixel in pixels
    )

    return chars


def build_ascii(chars, width):
    return "\n".join(
        chars[i:i + width]
        for i in range(0, len(chars), width)
    )


def main():

    image = Image.open(IMAGE_PATH)

    image = resize(image, WIDTH)

    image = gray(image)

    chars = convert(image)

    ascii_art = build_ascii(chars, WIDTH)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(ascii_art)

    print("ASCII generated successfully!")
    print(f"Saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
