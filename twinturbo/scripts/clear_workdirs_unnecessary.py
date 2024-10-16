import os

def delete_npy_files_in_cwola(start_directory):
    # Walk through the directory tree
    for root, dirs, files in os.walk(start_directory):
        # Check if the current folder is named "cwola"
        if os.path.basename(root) == "dope_3000":
            # Loop through files in the current "cwola" folder
            for file in files:
                if file.endswith(".npy"):
                    file_path = os.path.join(root, file)
                    try:
                        os.remove(file_path)  # Delete the .npy file
                        print(f"Deleted {file_path}")
                    except Exception as e:
                        print(f"Error deleting {file_path}: {e}")
                if file.endswith(".txt"):
                    file_path = os.path.join(root, file)
                    try:
                        os.remove(file_path)  # Delete the .npy file
                        print(f"Deleted {file_path}")
                    except Exception as e:
                        print(f"Error deleting {file_path}: {e}")

if __name__ == "__main__":
    # Replace this with the starting directory path you want to search in
    start_directory = "/home/users/o/oleksiyu/WORK/hyperproject/twinturbo/workspaces/"
    delete_npy_files_in_cwola(start_directory)