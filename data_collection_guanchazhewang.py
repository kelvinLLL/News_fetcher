import sys
import argparse
sys.path.append('/content/drive/MyDrive/Proj/News_fetcher')  # adjust path if needed

from data_collection_utilities import (
    # Variables
    prefix,
    suffix,
    
    # Web scraping utils
    wait_for_stable_page,
    safe_get,
    get_fresh_driver,
    restart_driver_and_get,

    # I/O
    save_lists_to_csv,
    load_csv,
    
    # Site-specific functions
    cankaoxiaoxi_index_visiting,
    cankaoxiaoxi_generalColumns_visiting,
    clean_contents_and_save
)

def main(index_url, other_url, prefix, file_name):

    sublinks = []
    contents = []
    htmls = []
    tags = []

    print("\n🌐 Visiting index page...")
    cankaoxiaoxi_index_visiting(sublinks, contents, htmls, url=index_url)

    print("\n💾 Saving index data...")
    save_lists_to_csv(file_name, sublink = sublinks, content = contents, html = htmls)

    if other_url:
      print("\n🌐 Visiting generalColumns page...")
      cankaoxiaoxi_generalColumns_visiting(
          sublinks, contents, htmls, tags, tag="generalColumns", col_url=other_url
      )
      print("\n💾 Saving other url data...")
      save_lists_to_csv("other_contents.csv", sublink = sublinks, content = contents, html = htmls, tag = tags)
    else:
      print("\n No other url input")
    
    print("\n🧹 Cleaning content...")
    print("\n🧹 First for Index Content")
    # index_contents = load_csv(file_name)
    cleaned_df = clean_contents_and_save(file_name)
    if other_url:
      print("\n🧹 Then for Other Content")
      clean_contents_and_save("other_contents.csv")
    print(f"\n✅ Done. Total cleaned articles in index: {len(cleaned_df)}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crawl and clean cankaoxiaoxi content.")

    parser.add_argument("--index_url", type=str, default="https://www.cankaoxiaoxi.com/#/index", help="URL for the homepage/index")
    parser.add_argument("--other_url", type=str, default="", help="URL for the generalColumns page") #https://www.cankaoxiaoxi.com/#/generalColumns/guandian
    parser.add_argument("--prefix", type=str, default="/content/drive/MyDrive/Proj/News_fetcher/", help="Output directory prefix")
    parser.add_argument("--file_name", type=str, default="index_contents.csv", help="Base CSV file name (e.g., index_contents.csv)")

    args = parser.parse_args()

    # Set prefix globally if needed
    prefix = args.prefix

    main(args.index_url, args.other_url, prefix, args.file_name)


