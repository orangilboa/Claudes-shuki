import os
import shutil

def read_file(path):
    with open(path, 'r') as f:
        return f.read()

def write_file(path, content):
    with open(path, 'w') as f:
        f.write(content)

def move_file(src_path, dst_path):
    """
    Move a file from the source path to the destination path.
    
    Args:
        src_path (str): The source file path.
        dst_path (str): The destination file path.
    """
    shutil.move(src_path, dst_path)

def delete_file(path):
    """
    Delete a file at the specified path.
    
    Args:
        path (str): The path of the file to delete.
    """
    os.remove(path)

def create_directory(path):
    """
    Create a new directory at the given path.
    
    Args:
        path (str): The path of the new directory to create.
    """
    os.makedirs(path, exist_ok=True)

def remove_directory(path):
    """
    Delete an existing directory at the given path.
    
    Args:
        path (str): The path of the directory to delete.
    """
    os.rmdir(path)
