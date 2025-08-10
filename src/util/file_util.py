import pathlib, os, shutil

def create_path(path: pathlib.Path):
  if not pathlib.Path.exists(path):
    pathlib.Path.mkdir(path, parents=True)
    
def remove_path(path: pathlib.Path):
  if pathlib.Path.exists(path):
    shutil.rmtree(path)
      
def num_of_items(path: pathlib.Path):
  if pathlib.Path.exists(path):
    for root, dirs, files in os.walk(path):
      return len(dirs) + len(files)
  return 0

def copy_files(files: list[pathlib.Path], dst: pathlib.Path, dst_map=None):  
  for file in files:
    if pathlib.Path.exists(file):
      if dst_map is not None:
        new_file_name = dst_map(pathlib.Path(file.name))
        shutil.copy(file, dst / new_file_name)
      else:
        shutil.copy(file, dst)
      
def clean_file_name(path: pathlib.Path):
  file_extension = path.suffix
  
  file_name = path.stem.replace('.', '')
  return path.parent / (file_name + file_extension)
  
      


# paths_to_remove = [path]
# stack = [path]

# while len(stack) != 0:
#   item = stack.pop()
  
#   for root, dirs, files in os.walk(item):
#     for file in files:
#       os.remove(item / file)
      
#     for dir in dirs:
#       stack.append(item / dir)
#       paths_to_remove.append(item / dir)
  
# for i in range(len(paths_to_remove)):
#   pathlib.Path.rmdir(paths_to_remove.pop())