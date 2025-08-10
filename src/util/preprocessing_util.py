import os, pathlib, math
from util import file_util

model_paths = ["test", "train", "validation"]

def get_classifications_with_file_count(start_path: pathlib.Path, max_depth: int = 1) -> list[pathlib.Path, int]:
  """
  Get classifications for data seperated by folders.
  
  Looks at all folders under the starting path, and uses their names for the classification of all files under them
  until a depth of n.
  Returns a list of tuples containing the classification, and list of filenames.
  """
  classifications = []
  stack = [(start_path, 0)] # index 0 = path, index 1 = depth

  while len(stack) != 0:
    path, current_depth = stack.pop()

    for root, dirs, files in os.walk(path):
      for dir in dirs:
        if current_depth != max_depth:
          stack.append((start_path / dir, current_depth+1))
          
      if len(files) != 0:
        classifications.append((path, files))
        
      break
  return classifications

def split_model_resources(src_path: pathlib.Path, 
                          dst_path: pathlib.Path, 
                          percentage_for_testing: int = 10, 
                          percentage_for_validation: int = 10, 
                          always_reload: bool = True):
  """
  Sort a list of categorized/uncategorized data into train, test, and validation data for use in model training.
  
  ## Arguments
  
  **src_path**
  - Path at which all unsorted data is stored.
  
  **dst_path**
  - Path at which all sorted data will be stored.
  
  **percentage_for_testing**
  - Percentage of unsorted data to allocate for testing.
  
  **percentage_for_validation**
  - Percentage of unsorted data to allocate for validation.
  
  **always_reload**
  - Whether to completely resort data every time this function is called.
  Default is True, if it is False, it will only reload data if it's folder is completely missing.
  """
  
  classifications = get_classifications_with_file_count(src_path, max_depth=0)
  
  for path, files in classifications:
    file_count = len(files)
    # Percentage of resources to split for testing and validation
    amount_for_testing = max(0, math.ceil((percentage_for_testing / 100) * file_count))
    amount_for_validation = max(0, math.ceil((percentage_for_validation / 100) * file_count))
    
    file_paths = list(map(lambda file: path / file, files)) # Map file names to relative paths
    
    files_for_training = file_paths[amount_for_testing+amount_for_validation:]
    files_for_testing = file_paths[:amount_for_testing]
    files_for_validation = file_paths[amount_for_testing:amount_for_testing+amount_for_validation]
    
    # Remove existing data in paths and create new.
    for model_path in model_paths:
      if always_reload:
        file_util.remove_path(dst_path / model_path)
      file_util.create_path(dst_path / model_path)
    
    file_util.copy_files(files_for_testing, dst_path / "test", dst_map=file_util.clean_file_name)
    file_util.copy_files(files_for_training, dst_path / "train", dst_map=file_util.clean_file_name)
    file_util.copy_files(files_for_validation, dst_path / "validation", dst_map=file_util.clean_file_name)