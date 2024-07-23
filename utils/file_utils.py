import requests
import urllib
import os
import pickle
import yaml
import glob
import shutil
from datetime import datetime, timedelta


def replace_extension(asset_path, new_extension):
    base_name = os.path.splitext(asset_path)[0].strip()
    extension = os.path.splitext(asset_path)[1].strip()

    return f"{base_name}{new_extension}"

### IMPORTANT FOR WINDOWS USERS TO SUPPORT LONG FILENAME PATHS 
### IN CASE YOU"RE USING LONG FILENAMES, AND THIS IS CAUSING AN EXCEPTION, FOLLOW THESE 2 STEPS:
# 1. change a registry setting to allow long path names on this particular Windows system (use regedit.exe): under HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem, set LongPathsEnabled to DWORD value 1
# 2. Check if the group policy setting is configured to enable long path names. Open the Group Policy Editor (gpedit.msc) and navigate to Local Computer Policy > Computer Configuration > Administrative Templates > System > Filesystem. Look for the "Enable Win32 long paths" policy and make sure it is set to "Enabled".
def write_to_file(text, text_filename, mode = 'a'):
    try:
        text_filename = text_filename.replace("\\", "/")
        with open(text_filename, mode, encoding='utf-8') as file:
            file.write(text)

        print(f"Writing file to full path: {os.path.abspath(text_filename)}")
    except Exception as e:
        logc(f"SERIOUS ERROR: {bc.RED}Error writing text to file: {e}{bc.ENDC}")

def read_asset_file(text_filename):
    try:
        text_filename = text_filename.replace("\\", "/")
        with open(text_filename, 'r', encoding='utf-8') as file:
            text = file.read()
        status = True
    except Exception as e:
        text = ""
        print(f"WARNING ONLY - reading text file: {e}")
        status = False

        

def read_yaml(yaml_path):
    # Read the YAML file
    with open(yaml_path, 'r') as file:
        settings = yaml.safe_load(file)
    return settings


def write_yaml(yaml_path, settings):
    # Write the changes back to the YAML file
    with open(yaml_path, 'w') as file:
        yaml.safe_dump(settings, file)



def copy_files(source_directory, destination_directory):
    # Ensure the destination directory exists
    os.makedirs(destination_directory, exist_ok=True)

    # List all files in the source directory
    files = os.listdir(source_directory)

    # Copy each file from the source to the destination directory
    for file_name in files:
        # Construct the full file path
        source_file = os.path.join(source_directory, file_name)
        destination_file = os.path.join(destination_directory, file_name)

        # Check if it's a file (and not a directory)
        if os.path.isfile(source_file):
            shutil.copy2(source_file, destination_file)  # copy2 preserves metadata
            print(f"Copied {file_name} to {destination_directory}")

    print("All files copied successfully.")




def copy_file(source_file, destination_directory):
    # Ensure the destination directory exists
    os.makedirs(destination_directory, exist_ok=True)

    destination_file = os.path.join(destination_directory, os.path.basename(source_file))

    # Check if it's a file (and not a directory)
    if os.path.isfile(source_file):
        shutil.copy2(source_file, destination_file)  # copy2 preserves metadata
        print(f"Copied {os.path.basename(source_file)} to {destination_directory}")

    print("All files copied successfully.")


def delete_folder(folder_path):
    # Check if the folder exists
    if os.path.exists(folder_path):
        # Delete the folder and all its contents
        shutil.rmtree(folder_path)
        print(f"Folder '{folder_path}' and all its contents have been deleted.")
    else:
        print(f"Folder '{folder_path}' does not exist.")


def delete_files_with_extension(folder_path, file_extension):
    # Construct the full pattern for glob
    pattern = os.path.join(folder_path, f'*{file_extension}')

    # Get the list of all files with the given extension
    files_to_delete = glob.glob(pattern)

    # Delete each file
    for file_path in files_to_delete:
        try:
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

    print("Deletion of files with the given extension completed.")