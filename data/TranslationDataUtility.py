import os
import requests
import zipfile
import shutil

class TranslationDataUtility:
    
    @staticmethod
    def extract_and_rename(url, target_name="fra-eng.txt"):
        zip_temp = "temp_data.zip"
        extract_dir = "temp_extract"
        
        print(f"Initiating download from: {url}")

        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(zip_temp, 'wb') as f:
                f.write(response.content)
        else:
            raise ConnectionError(f"Failed to reach server. Status: {response.status_code}")

        print("Extracting contents...")
        with zipfile.ZipFile(zip_temp, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

        source_path = os.path.join(extract_dir, 'fra-eng', 'fra.txt')
        
        if os.path.exists(source_path):
            shutil.move(source_path, target_name)
            print(f"Success: File is now available as '{target_name}'")
        else:
            print("Warning: Standard path not found. Searching for any .txt file...")
            for root, dirs, files in os.walk(extract_dir):
                for file in files:
                    if file.endswith(".txt"):
                        shutil.move(os.path.join(root, file), target_name)
                        break

        if os.path.exists(zip_temp): os.remove(zip_temp)
        if os.path.exists(extract_dir): shutil.rmtree(extract_dir)
        print("Workspace cleaned.")
