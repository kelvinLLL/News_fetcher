import argparse
import pandas as pd
from data_collection_utilities import (
    # Variables
    prefix,
    suffix,

    # I/O
    save_lists_to_csv,
    load_csv,

    # Sort part
    search_by_tag
)

def main(tag, file_path, top_k):
    # Load DataFrame
    df = load_csv(file_path)

    # Search
    result_df = search_by_tag(df, tag, top_k)

    # Print nicely
    pd.set_option('display.max_colwidth', 80)
    print(f"\n🔎 Found {len(result_df)} articles tagged with “{tag}”:\n")
    print(result_df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search news by tag from a tagged article CSV.")
    parser.add_argument("--prefix", type=str, default="/content/drive/MyDrive/Proj/News_fetcher/", help="Output directory prefix")        
    parser.add_argument("--tag", type=str, default="科技", help="The tag to search for (e.g., '科技')")
    parser.add_argument("--tagged_file", type=str, default="tagged_index_contents.csv", help="CSV file to load")
    parser.add_argument("--top_k", type=int, default=None, help="Max number of results to show")

    args = parser.parse_args()
    prefix = args.prefix
    main(args.tag, args.tagged_file, args.top_k)
