import datetime;
import json
import exiv2.image
import exiv2
import uuid

from datetime import timezone
from PIL import Image
from multiformats import multihash

prompt = "Hello w\"orld"

seed = 2122344
guidance_scale = 3.5
num_inference_steps = 28
width = 1024
height = 1024

# creating a image object (new image object) with
# RGB mode and size 200x200
image = Image.new(mode="RGB", size=(200, 200))


# Add metadata to the image
date_time = str(datetime.datetime.now(timezone.utc))
model_id = "flux1-dev-fp8"
model_multihash = "1220dc4a58f44c1ba335822aaf041b2d19483bfd12d5dc260f6fac403f7be5f33181"
model_license_url = "https://huggingface.co/black-forest-labs/FLUX.1-dev/resolve/main/LICENSE.md"
model_license_multihash = "1220b7a00498845420da83aad42857f69fbfcf731fd1efa6d1bb596a884f2f2cbf53"
model_author = "Black Forest Labs"
input_type= "txt2img"
metadata_json = json.dumps({
  "model": {
    "name": "flux",
    "id": model_id,
    "multihash": model_multihash,
    "license_url": model_license_url,
    "license_multihash": model_license_multihash,
    "author": model_author
  },
  "input": {
    "type": input_type,
    "prompt": prompt,
    "seed": seed,
    "cfg_scale": guidance_scale,
    "steps": num_inference_steps,
    "width": width,
    "height": height
  }})

folder = "./samples"
filename = f"{folder}/{uuid.uuid4()}"

image.save(filename+".webp", format="webp")
image.save(filename+".png", format="png")
image.save(filename+".jpg", format="jpeg")

filename = filename+".png"

with open(filename, 'rb') as file:
    file_data = file.read()

digest = multihash.digest(file_data, "sha2-256")
image_multihash = digest.hex()

image = exiv2.ImageFactory.open(filename)
image.readMetadata()
exif_data = image.exifData()
exif_data["Exif.Image.Make"] = model_author
exif_data["Exif.Image.Model"] = model_id
exif_data["Exif.Image.ImageDescription"] = prompt
exif_data["Exif.Photo.UserComment"] = metadata_json
exif_data["Exif.Image.DateTime"] = date_time
exif_data["Exif.Photo.DateTimeOriginal"] = date_time
exif_data["Exif.Image.Artist"] = model_id
exif_data["Exif.Image.ImageID"] = image_multihash
exif_data["Exif.Image.Copyright"] = model_license_url+";"+model_license_multihash
image.writeMetadata()
