import streamlit as st
import requests
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ======================
# SECURE CONFIGURATION
# ======================
API_KEY = st.secrets["AIzaSyBaJHRgIEu8oZXWXlczY1a5bIEap6QLU6A"]  # From Streamlit secrets
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
    response = requests.get(YOUTUBE_CHANNEL_URL, params=params)
    data = response.json()
    
    if not data.get("items"):
        return None
        
    stats = data["items"][0]["statistics"]
    return {
        "subscribers": int(stats.get("subscriberCount", 0)),
        "views": int(stats.get("viewCount", 0)),
        "description": data["items"][0]["snippet"].get("description", "")
    }

def find_similar_channels(target_channel_id, keywords):
    """Find channels with similar descriptions or titles"""
    target_data = get_channel_stats(target_channel_id)
    if not target_data:
        return []
    
    # Search related channels
    search_params = {
        "part": "snippet",
        "type": "channel",
        "q": " ".join(keywords),
        "maxResults": 20,
        "key": API_KEY
    }
    search_response = requests.get(YOUTUBE_SEARCH_URL, params=search_params).json()
    
    # Compare descriptions and titles
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

def get_video_stats(video_id):
    """Fetch video statistics and channel information"""
    params = {
        "part": "snippet,statistics",
        "id": video_id,
        "key": API_KEY
    }
    response = requests.get(YOUTUBE_VIDEO_URL, params=params)
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

