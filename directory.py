import os

class AxDirectory:
    @staticmethod
    def list_directories(directory): 
        directories = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))] 
        return directories

    @staticmethod
    def search_files_and_folders(directory):
        for root, dirs, files in os.walk(directory):
            print(f"현재 디렉토리: {root}")
            print("폴더:")
            for dir_name in dirs:
                print(f"  {dir_name}")
            print("파일:")
            for file_name in files:
                print(f"  {file_name}")
            print()

