import sys
import argparse
import getpass
sys.path.append('/content/drive/MyDrive/Proj/News_fetcher')  # adjust path if needed

from data_collection_utilities import (
    # Variables
    prefix,
    suffix,

    # I/O
    save_lists_to_csv,
    load_csv,
    
    # Tagging-specific functions
    tagging_and_rating
)
def load_csv(file_name):
    return pd.read_csv(f'{prefix}data/crawled_data/{suffix}_{file_name}', encoding='utf-8-sig')

def main(cleaned_file_name, model_name, api_key="", prefix=""):
    original_file_name = cleaned_file_name
    tagging_and_rating(original_file_name, model_name=model_name, api_key=api_key, prefix=prefix)


# ---------- CLI Entry Point ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run tagging model on cleaned articles.")
    parser.add_argument("--prefix", type=str, default="/content/drive/MyDrive/Proj/News_fetcher/", help="Output directory prefix")    
    parser.add_argument("--cleaned_file_name", type=str, default="index_contents.csv", help="Path to cleaned CSV file")
    parser.add_argument("--model_name", type=str, default="qwen-turbo", help="Name of the model to use (e.g., qwen-turbo)")
    prefix = args.prefix
    args = parser.parse_args()
    # Prompt user for API key (safer than using input)
    api_key = getpass.getpass("🔑 Please enter your DashScope API key: ")
    
    main(args.cleaned_file_name, args.model_name, api_key=api_key, prefix=args.prefix)

