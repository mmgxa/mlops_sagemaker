import base64

from PIL import Image

from io import BytesIO

def decode_base64_to_image(encoding: str) -> Image:
    content = encoding.split(";")[1]
    image_encoded = content.split(",")[1]
    return Image.open(BytesIO(base64.b64decode(image_encoded)))

