import streamlit as st
import requests
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ======================
# SECURE CONFIGURATION
# ======================
API_KEY = st.secrets["AIzaSyBaJHRgIEu8oZXWXlczY1a5bIEap6QLU6A"] 
# Corrected secret key name
YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"
YOUTUBE_VIDEO_URL = "https://www.googleapis.com/youtube/v3/videos"
YOUTUBE_CHANNEL_URL = "https://www.googleapis.com/youtube/v3/channels"

# ======================
# CORE FUNCTIONS
# ======================
def get_channel_stats(channel_id):
    """Fetch channel statistics and monetization status"""
    params = {
        "part": "statistics,snippet",
        "id": channel_id,
        "key": API_KEY
    }
    try:
        response = requests.get(YOUTUBE_CHANNEL_URL, params=params)
        response.raise_for_status()
        data = response.json()
        
        if not data.get("items"):
            return None
            
        stats = data["items"][0]["statistics"]
        return {
            "subscribers": int(stats.get("subscriberCount", 0)),
            "views": int(stats.get("viewCount", 0)),
            "description": data["items"][0]["snippet"].get("description", "")
        }
    except Exception as e:
        st.error(f"Error fetching channel stats: {e}")
        return None

def find_similar_channels(target_channel_id):
    """Find channels with similar descriptions"""
    target_data = get_channel_stats(target_channel_id)
    if not target_data:
        return []
    
    try:
        # Search related channels using YouTube's relatedToChannelId parameter
        search_params = {
            "part": "snippet",
            "type": "channel",
            "relatedToChannelId": target_channel_id,
            "maxResults": 20,
            "key": API_KEY
        }
        response = requests.get(YOUTUBE_SEARCH_URL, params=search_params)
        response.raise_for_status()
        search_response = response.json()
        
        channels = []
        for item in search_response.get("items", []):
            chan_id = item["snippet"]["channelId"]
            chan_data = get_channel_stats(chan_id)
            if chan_data:
                channels.append({
                    "id": chan_id,
                    "desc": chan_data["description"],
                    "title": item["snippet"]["title"]
                })
        
        if not channels:
            return []
        
        # Calculate similarity scores
        vectorizer = TfidfVectorizer(stop_words="english")
        descs = [target_data["description"]] + [c["desc"] for c in channels]
        tfidf = vectorizer.fit_transform(descs)
        similarities = cosine_similarity(tfidf[0:1], tfidf[1:]).flatten()
        
        # Return top 5 matches
        results = []
        for idx in np.argsort(similarities)[::-1][:5]:
            results.append({
                "channel_id": channels[idx]["id"],
                "title": channels[idx]["title"],
                "score": f"{similarities[idx]:.2%}"
            })
        return results
        
    except Exception as e:
        st.error(f"Error finding similar channels: {e}")
        return []

def get_video_stats(video_id):
    """Fetch video statistics and channel information"""
    params = {
        "part": "snippet,statistics",
        "id": video_id,
        "key": API_KEY
    }
    try:
        response = requests.get(YOUTUBE_VIDEO_URL, params=params)
        response.raise_for_status()
        data = response.json()
        
        if not data.get("items"):
            return None
        
        video_data = data["items"][0]
        channel_id = video_data["snippet"]["channelId"]
        channel_data = get_channel_stats(channel_id)
        
        return {
            "title": video_data["snippet"]["title"],
            "channel_id": channel_id,
            "channel_title": video_data["snippet"]["channelTitle"],
            "views": int(video_data["statistics"].get("viewCount", 0)),
            "likes": int(video_data["statistics"].get("likeCount", 0))
        }  # Fixed missing closing brace
    except Exception as e:
        st.error(f"Error fetching video stats: {e}")
        return None

# ======================
# STREAMLIT UI
# ======================
st.title("YouTube Research Toolkit 🔍")

# Sidebar for Channel Analysis
with st.sidebar:
    st.header("Channel Tools")
    channel_id = st.text_input("Enter Channel ID:")
    
    if st.button("Check Monetization"):
        if channel_id:
            with st.spinner("Analyzing channel..."):
                data = get_channel_stats(channel_id)
                if data:
                    st.subheader("💰 Monetization Status")
                    eligible = data["subscribers"] >= 1000
                    st.metric("Subscribers", f"{data['subscribers']:,}")
                    st.metric("Estimated Eligibility", 
                             "✅ Yes" if eligible else "❌ No",
                             help="Requires 1,000+ subscribers and 4,000 watch hours")
                else:
                    st.error("Invalid Channel ID")
    
    if st.button("Find Similar Channels"):
        if channel_id:
            with st.spinner("Searching similar channels..."):
                similar = find_similar_channels(channel_id)
                if similar:
                    st.subheader("📺 Similar Channels")
                    for chan in similar:
                        st.write(f"{chan['title']} (Score: {chan['score']})")
                        st.code(f"Channel ID: {chan['channel_id']}")

# Main Area for Video Analysis
st.header("Video Analyzer 🎥")
video_id = st.text_input("Enter Video ID:")
if video_id and st.button("Get Video Stats"):
    video_data = get_video_stats(video_id)
    if video_data:
        st.subheader("Video Statistics")
        st.write(f"Title: {video_data['title']}")
        st.write(f"Channel: {video_data['channel_title']}")
        st.write(f"Views: {video_data['views']:,}")
        st.write(f"Likes: {video_data['likes']:,}")


