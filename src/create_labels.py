import os

from settings import DATA_SETS


images_dir = DATA_SETS['sketchy_test']['images']
sketches_dir = DATA_SETS['sketchy_test']['sketches']

image_paths = set([path for path in os.listdir(images_dir) if os.path.isdir(os.path.join(images_dir, path))])
sketches_paths = set([path for path in os.listdir(sketches_dir) if os.path.isdir(os.path.join(sketches_dir, path))])

paths = image_paths.union(sketches_paths)

print(image_paths == sketches_paths)
print(paths == sketches_paths)
print(paths == image_paths)
