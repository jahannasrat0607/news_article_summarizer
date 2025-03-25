import streamlit as st
import requests

# FastAPI backend URL
API_URL = "http://127.0.0.1:8000/fetch_news"

st.title("📰 News Summarizer & Sentiment Analysis")
st.write("Enter a company name to get the latest news, sentiment, and summaries.")

# User input for company name
company_name = st.text_input("Enter Company Name (e.g., Tesla, Google, Apple)", "Tesla")

if st.button("Fetch News"):
    with st.spinner("Fetching news..."):
        # Call FastAPI to fetch news
        response = requests.get(f"{API_URL}?company={company_name}")

        if response.status_code == 200:
            news_data = response.json()

            if "Articles" in news_data:
                st.subheader(f"📰 Latest News on {news_data['Company']}")

                for idx, article in enumerate(news_data["Articles"]):
                    st.markdown(f"### {idx + 1}. {article['Title']}")
                    st.write(f"**Sentiment:** {article['Sentiment']}")
                    st.write(f"**Topics:** {', '.join(article['Topics'])}")
                    st.write(f"**Summary:** {article['Summary']}")
                    st.write(f"🔗 [Read Full Article]({article['Link']})")

                    # Play Hindi audio
                    if article["Audio"]:
                        st.audio(article["Audio"], format="audio/mp3")

                    st.markdown("---")

        else:
            st.error("❌ Failed to fetch news. Please try again.")

