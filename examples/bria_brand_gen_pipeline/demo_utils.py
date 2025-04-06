import requests
import base64
from IPython.display import Image as IPyImage, display
from io import BytesIO
import os
from PIL import Image, ImageDraw, ImageFont
import gc
import tempfile

def display_pil_image(img: Image.Image, resize: int = None):
    """
    Efficiently displays a PIL Image in a Jupyter Notebook with optional resizing.
    
    Args:
        img (PIL.Image.Image): The image to display.
        resize (int, optional): The desired width of the image. 
                                The height is automatically adjusted to maintain aspect ratio.
                                If None, the original size is used.
    """
    if not isinstance(img, Image.Image):
        raise ValueError("Input must be a PIL Image object")

    # Resize image while maintaining aspect ratio
    if resize:
        width, height = img.size
        aspect_ratio = height / width  # Maintain original aspect ratio
        new_size = (resize, int(resize * aspect_ratio))
        img.thumbnail(new_size, Image.LANCZOS)  # Resize in place

    # Save the processed image to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmpfile:
        img.save(tmpfile.name)
        temp_filename = tmpfile.name  # Store file path

    # Display using IPython.display.Image
    display(IPyImage(filename=temp_filename))

    # Explicitly delete the image object and force garbage collection
    del img
    gc.collect()

def add_title_to_image(image, text, font_path='ArialBold.ttf', font_size=20, text_color="black", padding=10):

    # Load the font
    if font_path:
        font = ImageFont.truetype(font_path, font_size)
    else:
        font = ImageFont.load_default()
    
    # Calculate the text size and position
    draw = ImageDraw.Draw(image)
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    image_width, image_height = image.size
    title_height = text_height + 2 * padding
    new_image_height = image_height + title_height
    
    # Create a new image with space for the title
    new_image = Image.new('RGB', (image_width, new_image_height), "white")
    new_image.paste(image, (0, title_height))
    
    # Draw the title on the new image
    draw = ImageDraw.Draw(new_image)
    text_position = ((image_width - text_width) // 2, padding)
    draw.text(text_position, text, font=font, fill=text_color)
    
    return new_image    

def display_images(image_list, title = "", line_width=10, resize = 350, font_size=15, return_img=False):
    # resize images while keeping aspect ratio
    resized_images = []
    for img in image_list:
        img.thumbnail((resize, resize))
        resized_images.append(img)

    image_list = resized_images
    # Calculate the total width and the maximum height of the concatenated image
    total_width = sum(image.width for image in image_list) + line_width * (len(image_list) - 1)
    max_height = max(image.height for image in image_list)
    
    # Create a new blank image with the calculated dimensions and fill it with white
    concatenated_image = Image.new('RGB', (total_width, max_height), 'black')
    
    # Paste each image into the new image
    x_offset = 0
    for image in image_list:
        concatenated_image.paste(image, (x_offset, 0))
        x_offset += image.width + line_width
    
    if title:
        concatenated_image = add_title_to_image(concatenated_image, title, font_size=font_size)
    # return concatenated_image
    display_pil_image(concatenated_image)

    if return_img:
        return concatenated_image

def return_images_from_urls(image_urls):
    images = []
    for image_url in image_urls:
        img_response = requests.get(image_url)
        img = Image.open(BytesIO(img_response.content))
        images.append(img)
    return images    

def pad_image_to_square(img):
    width, height = img.size
    max_dim = max(width, height)
    new_img = Image.new('RGB', (max_dim, max_dim), (0, 0, 0))
    new_img.paste(img, ((max_dim - width) // 2, (max_dim - height) // 2))
    return new_img

# Function to convert a PIL Image to a base64-encoded string
def pil_image_to_base64(pill_image):
    buffered = BytesIO()
    pill_image.save(buffered, format="JPEG")  # You can change the format if needed
    encoded_string = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return encoded_string    

# show mask overlayed on the image
def display_mask(mask, image):
    image = image.copy()
    mask = mask.resize(image.size)
    mask = mask.convert("RGBA")
    mask.putalpha(128)
    image.paste(mask, (0, 0), mask)
    display_images([image], f"Gen-Fill Mask Overlay")
