from requests import get
from zipfile import ZipFile
import os


download_path = "./Semester_Project/data"
#url = "https://murena.io/s/gZTkaw5HLrSoW9D/download/data.zip"
#url = "https://murena.io/s/AcwcKHebTJZPEMD/download/data2.zip"
url = "https://murena.io/s/yaowg829Nbp5pDy/download/data3.zip"


def download(url, download_path):
    r = get(url, stream=True)
    with open(download_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024*32):
            if chunk:
                f.write(chunk)


def extract_zip(file, extract_dir):
    with ZipFile(file, "r") as z:
        z.extractall(path=extract_dir)


def check_download_data(url, extract_dir, remove=True):
    print("Downloading data files from cloud..\n")
    print(url)
    download(url, extract_dir+".zip")
    print("Extracting data into" + extract_dir)
    extract_zip(extract_dir+".zip", extract_dir)
    print(flush=True)
    if remove:
        os.remove(extract_dir+".zip")


check_download_data(url, download_path)
