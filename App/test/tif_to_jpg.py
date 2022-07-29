from PIL import Image
import glob, os

# os.chdir("test")
count = 0
for file in glob.glob("*.tif"):
    print(file)
    img = Image.open(file)
    name = str(file).rstrip("*.tif")
    img.save("test_" + str(count) + ".jpg", "JPEG")
    count += 1
