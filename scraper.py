from datetime import datetime
import yfinance as yf
import pandas as pd
#for news scrape
import requests
from bs4 import BeautifulSoup
from newspaper import Article
import concurrent.futures #parallel processing

def get_stock_data(ticker):
    """
    Fetches the last x days of stock data

    """
    print(f"Fetching data for ticker: {ticker}")

    #1. Download data using yfinance
    stock = yf.Ticker(ticker)
    df = stock.history(period='10d', interval='1d')

    #2 Check if data is empty (invalid ticker)
    if df.empty:
        print(f"Error: No data found for ticker '{ticker}'")
        return None
    
    #clean up the data
    clean_df = df[["Close"]].copy()
    #reset index so "Data" becomes a column, not the index
    clean_df.reset_index(inplace=True)
    #Convert the Date to a simple string (YYY-MM-DD) removing the timezone
    clean_df["Date"] = clean_df["Date"].dt.strftime('%Y-%m-%d')
    return clean_df


#=====================================================================================
#======================================================================================

def get_article_content(url):
    """
    Visits a URL, downloads the article, and returns the top 500 characters of text.
    Returns "Error" if it can't scrape (paywalls, anti-bot, etc).
    """
    try:
        article = Article(url)
        article.download()
        article.parse()
        # We only want the beginning. FinBERT can't read unlimited pages.
        return article.text[:1800] 
    except:
        return None

#======================================================================================


def get_news(ticker):
    """
    Scrapes Headlines AND Article text from FinViz 
    """
    print(f"Fetching news for {ticker}... ")

    #1. Set up URL
    url = f"https://finviz.com/quote.ashx?t={ticker}"
    #finzwiz block python requests, pretend to be browser
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        #download website HTML
        respone = requests.get(url, headers=headers)

        #parse HTML using BeautifulSoup
        soup = BeautifulSoup(respone.content, "html.parser")

        #find the news table (finviz uses id="news-table")
        news_table = soup.find(id="news-table") 

        if not news_table:
            print("Could not find news table")
            return None
        
        #COLLECT LINKS (sequantial & fast)
        tasks = [] #Store date headline and link
        rows = news_table.find_all("tr")

        #40 articles
        limit = 40
        last_date = "Unknown"

        for row in rows[:limit]:
            if not row.a: continue # skip empty rows

            headline = row.a.text
            link = row.a["href"]

            # --- DATE PARSING LOGIC ---
            raw_timestamp = row.td.text.strip().split()
            if len(raw_timestamp) == 2:
                date_part = raw_timestamp[0]
                
                if date_part == "Today":
                    last_date = datetime.now().strftime("%Y-%m-%d")
                else:
                    try:
                        dt_obj = datetime.strptime(date_part, "%b-%d-%y")
                        last_date = dt_obj.strftime("%Y-%m-%d")
                    except:
                        last_date = date_part # Fallback if format changes

            #stor the task but dont download yet            
            tasks.append( (last_date, headline, link) )

        print(f"Found {len(tasks)} headlines. Downloading articles in parallel...")


        # --- PHASE 2: PARALLEL DOWNLOADING ---
        # We create a pool of 10 workers
        urls = [t[2] for t in tasks] # Extract just the URLs
        
        full_texts = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            # Map the function 'get_article_content' to every URL in the list
            # This runs them all at the same time.
            results = list(executor.map(get_article_content, urls))
            full_texts = results

        # --- PHASE 3: MERGE DATA ---
        parsed_data = []
        for i in range(len(tasks)):
            date, headline, link = tasks[i]
            content = full_texts[i]
            
            # Fallback to headline if download failed
            final_text = content if content else headline
            parsed_data.append([date, headline, final_text, link])

        print("Done!")
        
        return pd.DataFrame(parsed_data, columns=['Date', 'Headline', 'Full_Text', 'Link'])
    
    except Exception as e:
        print(f"Error scraping news: {e}")
        return None



#This part runs only if file is executed directly (not imported as a module)
if __name__ == "__main__":
   ticker = "AAPL"

   #get stock prices
   stock_df = get_stock_data(ticker)
   print("\n--- Stock Data ---")
   print(stock_df)

   #get news
   news_df = get_news(ticker)
   print("\n--- News Headlines ---")
   print(news_df)