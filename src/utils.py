import os

def list_images(folder: str):
    """Return a list of image file paths in a folder."""
    valid_exts = (".png", ".jpg", ".jpeg")
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(valid_exts)]
