
import shutil
import os

def erase_folder(folder_path):
    """
    Erases the specified folder and handles any exceptions.
    
    Parameters:
    folder_path (str): The path to the folder to be erased.
    """
    try:
        shutil.rmtree(folder_path)
        print(f"Folder '{folder_path}' erased successfully.")
        # main_placeholder.text(f"Folder erased successfully.")
    except Exception as e:
        print(f"Error erasing folder: {e}")
        # main_placeholder.text(f"Error erasing folder")
    # time.sleep(3) # Time for showing the erase status


def check_folder_exists(folder_path):
    return os.path.exists(folder_path)