import os
from logger.logger import LOGI, LOGE, LOGW
import zipfile
from concurrent.futures import ThreadPoolExecutor
import urllib.request


TAG = "DOWNLOADER"


def download_data(source_url: str, output_dir: str):
    LOGI(TAG, f"Downloading data from {source_url} to {output_dir}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    zip_path = _download_dataset(source_url, output_dir)

    _extract_dataset(zip_path, output_dir)
    
import sys

def _download_dataset(url: str, output_path: str):
    zip_path = os.path.join(output_path, "data.zip")
    
    try:
        # Ping the server to learn the exact total file size
        req_head = urllib.request.Request(url, method='HEAD')
        try:
            with urllib.request.urlopen(req_head) as response:
                total_size = int(response.getheader('Content-Length', 0))
        except Exception:
            total_size = 0

        initial_size = 0
        if os.path.exists(zip_path):
            initial_size = os.path.getsize(zip_path)
            # If our local size identically matches the server, skip entirely!
            if total_size > 0 and initial_size == total_size:
                LOGI(TAG, f"Dataset already fully downloaded at {zip_path}.")
                return zip_path
            elif initial_size > 0:
                LOGI(TAG, f"Found partial download ({initial_size / (1024*1024):.1f}MB). Attempting to resume...")

        req = urllib.request.Request(url)
        if initial_size > 0:
            req.add_header('Range', f'bytes={initial_size}-')

        with urllib.request.urlopen(req) as response:
            # If server ignored our Range header (returns 200 instead of 206 Partial Content), we must restart
            if initial_size > 0 and response.getcode() == 200:
                LOGW(TAG, "Server doesn't support resuming. Restarting download from scratch.")
                initial_size = 0

            # Only append ('ab') if we are actively resuming, else write new ('wb')
            mode = 'ab' if initial_size > 0 else 'wb'
            downloaded = initial_size
            
            with open(zip_path, mode) as f:
                while True:
                    # Download chunks stream
                    chunk = response.read(65536)
                    if not chunk:
                        break
                    
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0:
                        percent = min(100, int(downloaded * 100 / total_size))
                        bar_length = 40
                        filled_length = int(bar_length * min(downloaded, total_size) // total_size)
                        bar = '=' * filled_length + '-' * (bar_length - filled_length)
                        sys.stdout.write(f"\rDownloading data.zip: [{bar}] {percent}% ({downloaded / (1024*1024):.1f}MB / {total_size / (1024*1024):.1f}MB)")
                    else:
                        sys.stdout.write(f"\rDownloading data.zip: {downloaded / (1024*1024):.1f}MB")
                    sys.stdout.flush()
                    
        print() # Drop down one line when fully complete
        LOGI(TAG, f"Download properly completed to {zip_path}")
        return zip_path

    except Exception as e:
        print()
        LOGE(TAG, f"Failed to download: {e}")
        raise

def _extract_dataset(zip_path: str, output_dir: str):
    sentinel_file = os.path.join(output_dir, ".extraction_done")
    
    if os.path.exists(sentinel_file):
        LOGI(TAG, f"Dataset already extracted in {output_dir}. Skipping.")
        return
    os.makedirs(output_dir, exist_ok=True)

    max_workers = os.cpu_count()
    LOGI(TAG, f"Extracting {zip_path} using {max_workers} workers...")
    
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for file in zip_ref.namelist():
                    executor.submit(zip_ref.extract, file, output_dir)
        with open(sentinel_file, 'w') as f:
            f.write("done")
            
        LOGI(TAG, "Extraction complete")
    except Exception as e:
        LOGE(TAG, f"Extraction failed: {e}")
        raise  


if __name__ == "__main__":
    download_data("dummy_url", "data/raw")
