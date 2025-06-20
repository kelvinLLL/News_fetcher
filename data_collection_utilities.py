import pandas as pd
from datetime import datetime

from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    WebDriverException,
    TimeoutException,
    NoSuchElementException,
    InvalidSessionIdException
)
from bs4 import BeautifulSoup
import time
import google_colab_selenium as gs
import trafilatura

import sys
sys.path.append('/content/drive/MyDrive/Proj/News_fetcher')

from tqdm import tqdm
from openai import OpenAI
from time import sleep

# Vars
prefix = "/content/drive/MyDrive/Proj/News_fetcher/"


# Get today's date
suffix = datetime.today().strftime("%Y-%m-%d")


def wait_for_stable_page(driver, timeout=10, interval=1.5):
    """Waits for the whole text content to stop changing."""
    stable_counter = 0
    prev_text = ""
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            html = driver.page_source
            curr_text = html.strip()
        except:
            curr_text = ""

        if curr_text == prev_text and len(curr_text) > 0:
            stable_counter += 1
            if stable_counter >= 2:  # Stable twice in a row
                return curr_text
        else:
            stable_counter = 0
        prev_text = curr_text
        time.sleep(interval)
        # print(len(curr_text))
    return
    raise TimeoutException("detailsPage content did not stabilize in time.")
import time

def wait_for_stable_page(driver, timeout=20, interval=1.5, tolerance=0.05):
    """
    Waits for the whole text content to stop changing significantly.
    If the change between two versions is <1% for 3 times in a row, it's considered stable.
    """
    stable_counter = 0
    prev_text = ""
    start_time = time.time()

    def similarity_ratio(text1, text2):
        if not text1 or not text2:
            return 0.0
        len1, len2 = len(text1), len(text2)
        min_len = min(len1, len2)
        diff = abs(len1 - len2) + sum(c1 != c2 for c1, c2 in zip(text1, text2))
        return 1.0 - diff / max(len1, len2)

    while time.time() - start_time < timeout:
        try:
            html = driver.page_source
            curr_text = html.strip()
        except:
            curr_text = ""

        if prev_text and similarity_ratio(prev_text, curr_text) >= (1 - tolerance):
            stable_counter += 1
            if stable_counter >= 3:
                return curr_text
        else:
            stable_counter = 0

        prev_text = curr_text
        time.sleep(interval)
    return
    raise TimeoutException("Page content did not stabilize in time.")

def safe_get(driver, url, max_retries=5, wait_between=3):
    for attempt in range(max_retries):
        try:
            driver.get(url)
            return  # success
        except TimeoutException:
            print(f"[Attempt {attempt+1}] Timeout loading: {url}")
            time.sleep(wait_between)
        except (WebDriverException, InvalidSessionIdException) as e:
            print(f"[{attempt+1}] WebDriver error: {e}. Restarting driver...")
            try:
                driver.quit()
            except Exception:
                pass
            driver = get_fresh_driver()
            time.sleep(2)
    raise TimeoutException(f"Failed to load {url} after {max_retries} attempts.")

def get_fresh_driver():
    try:
        print("Initializing Chromedriver")
        driver = gs.Chrome()
        time.sleep(2)  # Let browser settle
        return driver
    except Exception as e:
        print("Driver failed to initialize:", e)
        return None




def restart_driver_and_get(url, retries=3):
    for attempt in range(retries):
        driver = get_fresh_driver()
        if driver is None:
            continue
        try:
            driver.get(url)
            return driver
        except WebDriverException as e:
            print(f"[{attempt+1}] Error loading {url}: {e}")
            try:
                driver.quit()
            except:
                pass
            time.sleep(2)
    print("All attempts failed.")
    return None


def save_lists_to_csv(file_name, **columns):
    """
    Save multiple equal-length lists to a CSV file.

    Parameters:
    - file_name: str, the output CSV file path
    - columns: keyword arguments where key is column name and value is a list of column data
    """
    # Get the length of the first column
    lengths = [len(col) for col in columns.values()]
    if not all(length == lengths[0] for length in lengths):
        raise ValueError("All input lists must have the same length.")

    # Create DataFrame and save
    df = pd.DataFrame(columns)
    df.to_csv(f'{prefix}data/crawled_data/{suffix}_{file_name}', index=True, encoding='utf-8-sig')
    print(f"CSV saved to {prefix}data/crawled_data/{suffix}_{file_name}")


def load_csv(file_name):
    return pd.read_csv(f'{prefix}data/crawled_data/{suffix}_{file_name}', encoding='utf-8-sig')


def cankaoxiaoxi_index_visiting(sublinks, contents, htmls, url="https://www.cankaoxiaoxi.com/#/index"):
    driver = restart_driver_and_get(url)
    wait_for_stable_page(driver)


    num_articles = len(driver.find_elements(By.CLASS_NAME, "clickPart"))
    original_url = driver.current_url
    # Create manual tqdm progress bar
    pbar = tqdm(total=num_articles, desc="Scraping index")
    cnt = 0
    while cnt < num_articles:
        try:
            # print("Enter Index Page")
            original_window = driver.current_window_handle
            cards = driver.find_elements(By.CLASS_NAME, "clickPart")
            # print(driver.current_url)
            cards[cnt].click()

            for handle in driver.window_handles:
                if handle != original_window:
                    # print("Switching!!")
                    driver.switch_to.window(handle)
                    break
            wait_for_stable_page(driver)
            article_url = driver.current_url
            sublinks.append(article_url)
            htmls.append(driver.page_source)
            # print(article_url)
            # Scrape the article
            soup = BeautifulSoup(driver.page_source, "lxml")
            # article = soup.find("div", class_="detailsPage")
            # print(article.get_text(strip=True) if article else "No article found")
            article = soup.get_text()
            print(driver.title)
            print(article)
            contents.append(article)

            # Go back to the list page
            driver.close()
            driver.switch_to.window(original_window)
            # print(driver.current_url)

        except Exception as e:
            print(e)
            try:
                # safe_get(driver, url)
                driver = restart_driver_and_get(url)
                wait_for_stable_page(driver)
            except:
                print(f"[{cnt}] Retry failed. Aborting.")
                break
        cnt += 1
        pbar.update(1)

    pbar.close()
    driver.quit()


from tqdm import tqdm

def cankaoxiaoxi_generalColumns_visiting(sublinks, contents, htmls, tags, tag, col_url="https://www.cankaoxiaoxi.com/#/generalColumns/guandian", target_class="templateModule", load_button_clicks=3):
    # driver = get_fresh_driver()
    # safe_get(driver, url)
    driver = restart_driver_and_get(col_url)
    wait_for_stable_page(driver)


    load_button = driver.find_elements(By.CLASS_NAME, "generalColumns-loadMore")
    if load_button:
      load_button = load_button[0]
      for _ in range(load_button_clicks):
        load_button.click()
        wait_for_stable_page(driver)
        load_button = driver.find_elements(By.CLASS_NAME, "generalColumns-loadMore")[0]
        # print(len(driver.find_elements(By.CLASS_NAME, target_class)))

    num_articles = len(driver.find_elements(By.CLASS_NAME, target_class))
    # Create manual tqdm progress bar
    pbar = tqdm(total=num_articles, desc="Scraping index")
    cnt = 0
    while cnt < num_articles:
        try:
            print("Enter Page")
            original_window = driver.current_window_handle
            cards = driver.find_elements(By.CLASS_NAME, target_class)
            print(driver.current_url)
            cards[cnt].click()

            for handle in driver.window_handles:
                if handle != original_window:
                    print("Switching!!")
                    driver.switch_to.window(handle)
                    break
            wait_for_stable_page(driver)
            article_url = driver.current_url
            sublinks.append(article_url)
            tags.append(tag)
            htmls.append(driver.page_source)

            # Scrape the article
            soup = BeautifulSoup(driver.page_source, "lxml")
            # article = soup.find("div", class_="detailsPage")
            # print(article.get_text(strip=True) if article else "No article found")
            article = soup.get_text()
            print(driver.title)
            print(article)
            contents.append(article)


            # Go back to the list page
            driver.close()
            driver.switch_to.window(original_window)
            # print(driver.current_url)

        except Exception as e:
            print(e)
            try:
                # safe_get(driver, url)
                driver = restart_driver_and_get(url)
                wait_for_stable_page(driver)
            except:
                print(f"[{cnt}] Retry failed. Aborting.")
                break
        cnt += 1
        pbar.update(1)

    pbar.close()
    driver.quit()


def clean_contents_and_save(original_file_name):
    df = load_csv(original_file_name)
    # Extract clean text from each HTML
    clean_texts = df['html'].apply(
        lambda x: trafilatura.extract(x, include_comments=False, include_tables=False)
        if isinstance(x, str) and x.strip() else None
    )

    # Drop 'content' and 'html' columns, then add clean_text
    cleaned_df = df.drop(columns=['content', 'html'], errors='ignore').copy()
    cleaned_df['clean_text'] = clean_texts
    # Drop rows with less than 15 tokens in 'clean_text'
    cleaned_df = cleaned_df[cleaned_df['clean_text'].apply(
        lambda x: len(x) >= 15
    )].reset_index(drop=True)

    # Save the cleaned DataFrame to a new CSV file
    # save_lists_to_csv("index_contents.csv", sublink = sublinks, content = contents, html = htmls, tag = tags)
    cleaned_df.to_csv(f'{prefix}data/crawled_data/{suffix}_cleaned_{original_file_name}', index=False, encoding='utf-8-sig')
    print(f"CSV saved to {prefix}data/crawled_data/{suffix}_cleaned_{original_file_name}")
    return cleaned_df


# ==============================================================================================================================================================================
# Following is for LLM tagging
def make_prompt_cn(news):
    return f"""你是一个新闻分类助手，请根据以下新闻内容为其生成几个合适的主题标签。 你可以从以下标签中选择也可以自己生成合适的标签：政治、体育、财经、科技、健康、教育、国际、娱乐、环境、社会。

新闻内容：
\"\"\"{news}\"\"\"

请输出中文标签，以井号分隔：
标签：
"""


# Prompt builder
def make_tag_prompt(article):
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": make_prompt_cn(article)},
    ]

def make_rating_followup(messages_so_far, tag_response):
    return messages_so_far + [
        {"role": "assistant", "content": tag_response},
        {"role": "user", "content": "请你给这篇新闻的质量打分，满分10分。直接输出数字"},
    ]

def tagging_and_rating(original_file_name, model_name="qwen-turbo", api_key="", prefix=""):
    df = load_csv(f"cleaned_{original_file_name}")
    articles = df["clean_text"].tolist()[:]  # limit for testing
    urls = df["sublink"].tolist()[:]

    # Setup Qwen API client
    client = OpenAI(
        api_key=api_key,  # or api_key="sk-xxx"
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    # Store results
    results = []

    for i, article in enumerate(articles, start=1):
        try:
            # Step 1: Ask for tags
            messages = make_tag_prompt(article)
            tag_response = client.chat.completions.create(
                model=model_name,
                messages=messages,
            )
            tag_text = tag_response.choices[0].message.content
            tag_usage = tag_response.usage

            # Step 2: Ask for rating (continue chat)
            rating_messages = make_rating_followup(messages, tag_text)
            rating_response = client.chat.completions.create(
                model=model_name,
                messages=rating_messages,
            )
            rating_text = rating_response.choices[0].message.content
            rating_usage = rating_response.usage

            # Combine token usage
            total_input_tokens = tag_usage.prompt_tokens + rating_usage.prompt_tokens
            total_output_tokens = tag_usage.completion_tokens + rating_usage.completion_tokens
            total_tokens = tag_usage.total_tokens + rating_usage.total_tokens

            # Save result
            results.append({
                "index": i,
                "article": article,
                "tags": tag_text,
                "rating": rating_text,
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
                "total_tokens": total_tokens,
                "url": urls[i-1],
            })

            print(f"[{i}/{len(articles)}] ✅ tagged & rated: input {total_input_tokens}, output {total_output_tokens}")

        except Exception as e:
            print(f"[{i}] ❌ Error: {e}")
            results.append({
                "index": i,
                "article": article,
                "tags": None,
                "rating": None,
                "input_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
                "url": None,
            })
            sleep(2)  # optional cooldown

    # Save to DataFrame and CSV
    results_df = pd.DataFrame(results)
    # results_df.to_csv("tagged_articles_with_tokens.csv", index=False, encoding="utf-8-sig")
    results_df.to_csv(f'{prefix}data/crawled_data/{suffix}_tagged_{original_file_name}', index=False, encoding='utf-8-sig')
    # Ensure numeric type and ignore None values
    input_total = pd.to_numeric(results_df['input_tokens'], errors='coerce').sum()
    output_total = pd.to_numeric(results_df['output_tokens'], errors='coerce').sum()
    print(f"🔽 saved to {prefix}data/crawled_data/{suffix}_tagged_{original_file_name}")

    print(f"\n📊 Token Cost Summary:")
    print(f"🟦 Total input tokens:  {int(input_total)}")
    print(f"🟨 Total output tokens: {int(output_total)}")


# ==============================================================================================================================
# Following is for sort and app part

def search_by_tag(df, tag, top_k=None):
    """
    Return news containing the specified tag, sorted by rating descending.
    """
    required_cols = ['tags', 'article', 'rating', 'url']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: '{col}'")

    # Filter rows containing the tag
    filtered = df[df['tags'].apply(lambda x: isinstance(x, str) and tag in x)]

    # Ensure rating is numeric
    filtered['rating'] = pd.to_numeric(filtered['rating'], errors='coerce')

    # Sort by rating descending
    filtered = filtered.sort_values(by='rating', ascending=False)

    # Limit to top_k if provided
    if top_k:
        filtered = filtered.head(top_k)

    return filtered[['url', 'article', 'rating']].reset_index(drop=True)

