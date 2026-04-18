import argparse
from config import CONFIG
from preprocessing.download import download_data
from preprocessing.prepare import prepare_data
from preprocessing.cleaner import remove_raw_data, remove_clean_data, remove_all, remove_zip


def main():
    parser = argparse.ArgumentParser(description="RadBrecim Data Preparation Pipeline")
    parser.add_argument("--clean-raw", action="store_true", help="Remove extracted raw data (keeps the zip file)")
    parser.add_argument("--clean-zip", action="store_true", help="Remove the downloaded zip file (forces re-download)")
    parser.add_argument("--clean-prepared", action="store_true", help="Remove prepared/cleaned data images")
    parser.add_argument("--clean-all", action="store_true", help="Complete cleanup of all data (zip, raw, and clean)")
    args = parser.parse_args()

    # Execute cleaning tasks if requested
    if args.clean_all:
        remove_all(CONFIG.raw_dir, CONFIG.clean_dir)
    else:
        if args.clean_raw:
            remove_raw_data(CONFIG.raw_dir)
        if args.clean_zip:
            remove_zip(CONFIG.raw_dir)
        if args.clean_prepared:
            remove_clean_data(CONFIG.clean_dir)

    # Continue with normal pipeline (will bypass automatically if already done)
    download_data(CONFIG.source_url, CONFIG.raw_dir)
    prepare_data(CONFIG.raw_dir, CONFIG.clean_dir, CONFIG.classes_raw_dirs)

if __name__ == "__main__":
    main()
