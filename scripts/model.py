from PIL import Image
import io

def get_shape_img(img_byte):
    image = Image.open(io.BytesIO(img_byte))
    return image.size

    