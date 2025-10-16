from langchain_core.messages import HumanMessage

from PIL import Image
import os
import base64

def ensure_supported_format(image_path: str) -> str:
    """
    Ensures the image is in JPG or PNG format.
    Converts WEBP (or other formats) to JPG automatically.
    Returns the path to the safe file.
    """
    ext = os.path.splitext(image_path)[1].lower()
    if ext in [".jpg", ".jpeg", ".png"]:
        return image_path  # already safe format
    
    # Convert unsupported formats (e.g., .webp) to .jpg
    img = Image.open(image_path).convert("RGB")
    safe_path = image_path.rsplit(".", 1)[0] + ".jpg"
    img.save(safe_path, "JPEG")
    return safe_path

def image_to_base64(image_path: str) -> str:
    """Reads a local image and returns a base64-encoded data URL."""
    with open(image_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode("utf-8")
    ext = os.path.splitext(image_path)[1].lower().replace(".", "")
    if ext == "jpg":
        ext = "jpeg"  # correct MIME type
    return f"data:image/{ext};base64,{b64}"

def describe_image_with_langchain(
    llm,
    image_path: str,
    detail_level: str = "very detailed",
    item: str = "quilt",
    local_img: bool = True
):
    """
    Describe an image using an LLM via LangChain multimodal input.
    """

    if local_img:
        # Ensure we have a supported format
        safe_image_path = ensure_supported_format(image_path)

        # Encode local file as base64 data URL
        data_url = image_to_base64(safe_image_path)
    else:
        data_url = image_path

    # Build multimodal input
    message = HumanMessage(
        content=[
            {
                "type": "text",
                "text": (
                    f"Describe the {item} in {detail_level} for Midjourney without command, "
                    f"mentioning all visible niche, objects, colors, context, vibe and actions. "
                    f"Return the design only, no special character such as *, -. "
                    f"Example: a stunning quilt bedding set features a vibrant tree of Life design "
                    f"that blends intricate stitching and vibrant colors to evoke a sense of nature's "
                    f"beauty and harmony."
                ),
            },
            {"type": "image_url", "image_url": {"url": data_url}},
        ]
    )

    # Call LLM
    response = llm.invoke([message])
    return response.content

def tagging_image_with_langchain(llm, image_path: str, local_img = True):

    if local_img:    
        # Ensure we have a supported format
        safe_image_path = ensure_supported_format(image_path)
        
        # Encode local file as base64 data URL
        data_url = image_to_base64(safe_image_path)
    else:
        data_url = image_path
    
    # Build multimodal input
    message = HumanMessage(
        content=[
            {"type": "text", "text": f"Describe this design in detailed tags, including: niche, color, vibe, product type, design elements, theme. Return the result in raw JSON format, without code fences"},
            {"type": "image_url", "image_url": {"url": data_url}}
        ]
    )
    
    # Call GPT
    response = llm.invoke([message])
    return response.content

