from requests import get
from zipfile import ZipFile
import os


download_path = "./data"
url = "https://murena.io/s/XegtijPDxW2oPsW"


def download(url, download_path):
    r=get(url,stream=True)
    with open(download_path,"wb") as f:
        for chunk in r.iter_content(chunk_size=1024*32):
            if chunk:
                f.write(chunk)

def extract_zip(file, extract_dir):
    with ZipFile(file,"r") as z:
        z.extractall(path = extract_dir)  


def check_download_data(url, extract_dir, remove=True):
    print("Downloading data files from cloud..\n")
    print(url)
    download(url, extract_dir+".zip")
    print("Extracting data into" + extract_dir)
    extract_zip(extract_dir+".zip", os.path.dirname(extract_dir))
    print(flush=True)
    if remove:
        os.remove(extract_dir+".zip")
         
