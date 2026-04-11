import json
import requests
from tqdm import tqdm
import pandas as pd


def ingest_page_video_ids(response: requests.models.Response) -> list:
    """Ingest YouTube video IDs from a Youtube channel's individual page.
    The API will only return the first 50 video so we will need to iterate
    through several "pages" to get all the video IDs for a given channel.

    Args:
        response (requests.models.Response): Content of an API response.

    Returns:
        list: Basic video information for a given channel's API request. This
        information includes the channel ID (channel_id) along with each video's
        ID (video_id), publish date & time (datetime), and title (title).
    """

    page_video_id_list = []

    for raw_item in json.loads(response.text)["items"]:
        # only execute for youtube videos
        if raw_item["id"]["kind"] != "youtube#video":
            continue

        video_record = {}
        video_record["channel_id"] = raw_item["snippet"]["channelId"]
        video_record["video_id"] = raw_item["id"]["videoId"]
        video_record["datetime"] = raw_item["snippet"]["publishedAt"]
        video_record["title"] = raw_item["snippet"]["title"]

        page_video_id_list.append(video_record)

    return page_video_id_list


def ingest_video_ids(api_key):
    url = "https://www.googleapis.com/youtube/v3/search"
    video_id_list = []
    
    queries = ["a", "the", "music", "review", "funny", "news"]
    
    for query in queries:
        page_token = None

        while True:
            params = {
                "key": api_key,
                "q": query,
                "part": ["snippet", "id"],
                "type": "video",
                "order": "date",
                "maxResults": 50,
            }

            if page_token:
                params["pageToken"] = page_token

            response = requests.get(url, params=params)
            data = json.loads(response.text)

            # DEBUG: print API errors
            if "items" not in data:
                print("\n--- API ERROR ---")
                print("Query:", query)
                print("Params:", params)
                print("Response:", data)
                break

            video_id_list += ingest_page_video_ids(response)

            page_token = data.get("nextPageToken")
            if not page_token:
                break

    return video_id_list

# def ingest_video_ids(api_key, page_token=None):
#     """Ingest all YouTube video IDs for a given Youtube channel.

#     Args:
#         api_key (str): Your Youtube API key.
#         channel_id (str): Youtube channel ID.
#         page_token (str, optional): Each API request will only return a max of 50
#         results. To get all results, you must go to the next 'page'. page_token
#         allows you to pull results from a single page. Setting to the defaults of
#         None will start at the first page and then iterate through all available
#         pages.

#     Returns:
#         list: Basic video information for a given channel's API request. This
#         information includes the channel ID (channel_id) along with each video's
#         ID (video_id), publish date & time (datetime), and title (title).
#     """
#     # base URL
#     url = "https://www.googleapis.com/youtube/v3/search"

#     # intialize list to store video data
#     video_id_list = []
    
#     queries = ["a", "the", "music", "review", "funny", "news"]
    
#     for query in queries:
#         page_token = None

#         # extract video data across multiple search result pages
#         while True:
#             # define parameters for API call
#             params = {
#                 "key": api_key,
#                 "q": query,
#                 "part": ["snippet", "id"],
#                 "order": "date",
#                 "maxResults": 50,
#             }
#             # make get request
#             response = requests.get(url, params=params)
    
#             # append video records to list
#             video_id_list += ingest_page_video_ids(response)
    
#             try:
#                 # grab next page token
#                 page_token = json.loads(response.text)["nextPageToken"]
#             except:
#                 # if no next page token kill while loop
#                 page_token = 0

#     return video_id_list


# def ingest_video_stats(video_id_data, api_key):
#     """Ingest basic statistics (i.e. number of views, likes & comments) for
#     Youtube videos.

#     Args:
#         video_id_data (list): List of video ID data from `ingest_channel_video_ids()`
#         api_key (_type_): Your Youtube API key.

#     Returns:
#         list: Same basic video information as ingest_channel_video_ids() but enhanced
#         with the count of each video's views, likes, and comments.
#     """
#     num_iterations = len(video_id_data)
#     for index, video in tqdm(
#         enumerate(video_id_data),
#         total=num_iterations,
#         bar_format="[{elapsed}<{remaining}] {n_fmt}/{total_fmt} | {l_bar}{bar} {rate_fmt}{postfix}",
#     ):
#         url = f"https://www.googleapis.com/youtube/v3/videos?id={video['video_id']}&key={api_key}&part=statistics"
#         response = requests.get(url)
#         stats = json.loads(response.text)["items"][0]["statistics"]

#         video_id_data[index]["views"] = stats["viewCount"]
#         video_id_data[index]["likes"] = stats["likeCount"]
#         video_id_data[index]["comments"] = stats["commentCount"]

#     return video_id_data

def ingest_video_stats(video_ids, api_key):
    num_iterations = len(video_ids)

    for index, video_id in tqdm(
        enumerate(video_ids),
        total=num_iterations,
        bar_format="[{elapsed}<{remaining}] {n_fmt}/{total_fmt} | {l_bar}{bar} {rate_fmt}{postfix}",
    ):
        url = f"https://www.googleapis.com/youtube/v3/videos?id={video_id}&key={api_key}&part=statistics"
        response = requests.get(url)
        data = json.loads(response.text)

        stats = data["items"][0]["statistics"]

        video_ids[index] = {
            "video_id": video_id,
            "views": stats.get("viewCount"),
            "likes": stats.get("likeCount"),
            "comments": stats.get("commentCount"),
        }

    return video_ids




def get_all_comments_for_video(api_key, video_id):
    url = "https://www.googleapis.com/youtube/v3/commentThreads"
    comments = []
    page_token = None

    while True:
        params = {
            "key": api_key,
            "videoId": video_id,
            "part": "snippet,replies",
            "maxResults": 100,
            "textFormat": "plainText"
        }

        if page_token:
            params["pageToken"] = page_token

        response = requests.get(url, params=params)
        data = json.loads(response.text)

        # Stop if API returns an error
        if "items" not in data:
            print("Error for video:", video_id, data)
            break

        # Extract comments
        for item in data["items"]:
            top = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "video_id": video_id,
                "comment": top["textDisplay"]
            })

            # Extract replies if present
            if "replies" in item:
                for reply in item["replies"]["comments"]:
                    rep = reply["snippet"]
                    comments.append({
                        "video_id": video_id,
                        "comment": rep["textDisplay"]
                    })

        # Pagination
        page_token = data.get("nextPageToken")
        if not page_token:
            break

    return comments



def ingest_comments_for_videos(api_key, video_ids):
    all_comments = []

    for vid in video_ids:
        print("Pulling comments for:", vid)
        video_comments = get_all_comments_for_video(api_key, vid)
        all_comments.extend(video_comments)

    # Convert to DataFrame with exactly the columns you want
    df = pd.DataFrame(all_comments, columns=["video_id", "comment"])
    return df