import imageio
import glob

images = []

for image_path in glob.glob("ant_world_image_databases/routes/ant1_route1/*.png"):
    image = imageio.imread(image_path)
    images.append(image)

print(len(images))