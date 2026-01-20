import streamlit as st
import pandas as pd
import scraper
import analyzer
import plotly.express as px

#to run: streamlit run app.py

# 1. Page Config
st.set_page_config(page_title="Market Watchdog")

# 2. The Title
st.title("Stock News Sentiment Analyzer")
st.markdown("Analyze the latest news for any stock ticker using **FinBERT**.")

# 3. Sidebar
with st.sidebar:
    st.header("Search Settings")
    ticker = st.text_input("Enter Stock Ticker:", value="GOOG").upper()
    analyze_btn = st.button("Analyze Stock")

# 4. Main Logic
if analyze_btn:
    with st.spinner(f"Fetching data for {ticker}..."):
        # A. Get Data
        stock_df = scraper.get_stock_data(ticker)
        news_df = scraper.get_news(ticker)

        if stock_df is None or news_df is None:
            st.error("Could not fetch data. Check the ticker symbol.")
        else:
            # --- PART 1: STOCK PRICES ---
            st.subheader(f"{ticker} Stock Price (last 10 days)")
            fig = px.line(stock_df, x='Date', y='Close', markers=True)
            st.plotly_chart(fig)

            # --- PART 2: NEWS ANALYSIS ---
            st.subheader("Latest News Sentiment")
            
            # Use 'Full_Text' for the AI, but we will display 'Headline' later
            texts_to_analyze = news_df['Full_Text'].tolist()
            
            with st.spinner("Reading articles & Running AI..."):
                scores = analyzer.analyze_sentiment(texts_to_analyze)

            # Add results to DataFrame
            news_df['Sentiment'] = [s['label'] for s in scores]
            news_df['Confidence'] = [s['score'] for s in scores]

            # --- PART 3: CALCULATE TOTAL SCORE ---
            # Logic: Positive = +1, Negative = -1, Neutral = 0.
            # Weighted by confidence.
            total_score = 0
            for s in scores:
                if s['label'] == 'positive':
                    total_score += s['score']
                elif s['label'] == 'negative':
                    total_score -= s['score']
            
            # --- PART 3: CALCULATE TOTAL SCORE ---
            
            # Average score
            avg_score = total_score / len(scores)

            # Create the Explanation Text
            explanation = """
            **How the Score is Calculated:**
            Each article gets "points" based on the AI's confidence (0 to 1):
            
            * **Positive:** +1 x Confidence (e.g., 90% sure = **+0.9**)
            * **Negative:** -1 x Confidence (e.g., 80% sure = **-0.8**)
            * **Neutral:** 0 points (Drags the average toward zero)
            
            The final score is the **Average** of these points across all articles.
            
            *Example:* Article 1: Positive (+0.9)
            Article 2: Neutral (0.0)
            Article 3: Negative (-0.5)
            --------------------------
            Average = (+0.4) / 3 = **0.13** (Slightly Bullish)
            """

            # --- CUSTOM UI REPLACEMENT ---
            # We use columns to make it look like a metric
            col1, col2 = st.columns([1, 4])
            
            with col1:
                st.metric("Score", f"{avg_score:.2f}", help=explanation)
            
            with col2:
                # Custom HTML to show the label with correct color & arrow
                if avg_score > 0:
                    st.markdown(
                        "<h3 style='color:green; margin-top:0px;'>Bullish</h3>", 
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        "<h3 style='color:red; margin-top:0px;'>Bearish</h3>", 
                        unsafe_allow_html=True
                    )

            # Add the caption below
            st.caption(f"Based on {len(news_df)} articles. Scale ranges from -1 (Neg) to +1 (Pos).")
            # --- PART 4: DISPLAY THE COLORED TABLE ---
            # 1. Clean up for display (Round confidence)
            news_df['Confidence'] = news_df['Confidence'].round(2)
            
            # 2. Select columns to show
            display_df = news_df[['Sentiment', 'Confidence', 'Headline', 'Date']]

            # 3. Color styling function
            def color_sentiment(val):
                color = 'black'
                if val == 'positive': color = 'green'
                elif val == 'negative': color = 'red'
                elif val == "neutral": color = "white"
                return f'color: {color}; font-weight: bold'

            # 4. Show the table
            st.dataframe(
                display_df.style.map(color_sentiment, subset=['Sentiment']),
                width="stretch",
                hide_index=True
            )
            
            # --- PART 5: SUMMARY ---
            positive_count = news_df[news_df['Sentiment'] == 'positive'].shape[0]
            negative_count = news_df[news_df['Sentiment'] == 'negative'].shape[0]
            neutral_count = news_df[news_df['Sentiment'] == 'neutral'].shape[0]

            # Calculate the "Net Sentiment" (ignoring neutrals)
            # If we have 0 positive and 0 negative, it's truly neutral.
            if positive_count == 0 and negative_count == 0:
                st.info("Overall Consensus: QUIET / NEUTRAL (No strong signals)")
            
            elif positive_count > negative_count:
                ratio = positive_count / (positive_count + negative_count)
                # If it's a landslide (e.g., 5 Pos vs 1 Neg)
                if ratio > 0.7:
                    st.success(f"Overall Consensus: BULLISH ({positive_count} vs {negative_count})")
                else:
                    st.success(f"Overall Consensus: LEANS BULLISH ({positive_count} vs {negative_count})")
            
            elif negative_count > positive_count:
                ratio = negative_count / (positive_count + negative_count)
                if ratio > 0.7:
                    st.error(f"Overall Consensus: BEARISH ({negative_count} vs {positive_count})")
                else:
                    st.error(f"Overall Consensus: LEANS BEARISH ({negative_count} vs {positive_count})")
            
            else:
                # Exact tie (e.g., 2 Pos vs 2 Neg)
                st.info(f"Overall Consensus: MIXED / UNCERTAIN ({positive_count} vs {negative_count})")



            # Proof of Work
            help_text = """
            You can view the raw article text analyzed by the AI by
            double clicking one of the full text cells.
            
            If only the headline is shown, it means the article probably had a paywall 
            and could not be downloaded.
            """
            
            with st.expander("View Raw Article Text (Proof of Analysis)"):
                st.info(help_text) 
                
                # Then show the dataframe

                st.dataframe(news_df[['Headline', 'Full_Text']].head(5))
