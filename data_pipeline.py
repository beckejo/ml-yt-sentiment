import great_expectations as gx
import os
import numpy as np
import pandas as pd
import unicodedata
import warnings
import re
from datetime import datetime

from dataops_utils import (
    ingest_video_ids,
    ingest_video_stats,
    get_all_comments_for_video,
    ingest_comments_for_videos
)

# Define API Keys
VIDEOS_API_KEY = 'AIzaSyAK7LjAPNQkvoTZqcpd1zTllbITBIBe5kw'
COMMENTS_API_KEY = 'AIzaSyBmiNrgUJsUYDrd1XzPP6EEfWJfR1QBt_A'
STATS_API_KEY = 'AIzaSyA0GVVhtbbPrR9hiqiTxRiP8YZ8HwChedU'

# Define URL
BASE_URL = "https://www.googleapis.com/youtube/v3"

# Search for videos and pull list of video_ids
video_ids = ingest_video_ids(VIDEOS_API_KEY)
video_ids = pd.DataFrame(video_ids)
video_ids_list = list(video_ids['video_id'])

# Serach for video stats from videos returned in video search
stats = ingest_video_stats(video_ids_list, STATS_API_KEY)
stats = pd.DataFrame(stats)

# Convert datatypes
stats['views'] = stats['views'].astype('Int64')
stats['likes'] = stats['likes'].astype('Int64')
stats['comments'] = stats['comments'].astype('Int64')

# Only include videos with more then 10 comments
stats = stats[stats['comments'] >= 10]

# Create target variable: likes per 100 views
# Bucket that into 3 groups based on percentiles as sentiment outcome
stats['likes_per_100_views'] = round(stats['likes'] / stats['views'] * 100, 3)
stats = stats[stats['likes_per_100_views'].notna()]
stats['sentiment'] = pd.qcut(stats['likes_per_100_views'], q = 3, labels = [0, 1, 2])
stats['sentiment'] = stats['sentiment'].astype('Int64')

# Create distinct list of filtered video ids to use for comment search
videos_id_for_comments = list(stats['video_id'])

# Ingest video comments
comments = ingest_comments_for_videos(COMMENTS_API_KEY, videos_id_for_comments)

# Concatenate all comments into one record per video
comments = comments.groupby('video_id')['comment'].agg(lambda x: ' '.join(x)).reset_index()

# Clean comment text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)          # remove URLs
    text = re.sub(r"[^a-zA-Z\s]", "", text)      # remove punctuation/numbers
    text = re.sub(r"\s+", " ", text).strip()     # normalize whitespace
    return text

comments["clean_comment"] = comments["comment"].apply(clean_text)

# Complete data: join together stats and comments
complete_data = stats.merge(comments, how = 'inner', on = 'video_id')



# Create Data Context.
context = gx.get_context()

# Create Data Source, Data Asset, Batch Definition, and Batch.
data_source = context.data_sources.add_pandas("pandas")
data_asset = data_source.add_dataframe_asset(name="Youtube video data")
batch_definition = data_asset.add_batch_definition_whole_dataframe("batch definition")
batch = batch_definition.get_batch(batch_parameters={"dataframe": complete_data})



# Create an Expectation Suite
suite = gx.ExpectationSuite(name="Youtube video data expectations")

# Add the Expectation Suite to the Data Context
suite = context.suites.add(suite)

# Validate columns exist
suite.add_expectation(gx.expectations.ExpectColumnToExist(column='video_id'))
suite.add_expectation(gx.expectations.ExpectColumnToExist(column='views'))
suite.add_expectation(gx.expectations.ExpectColumnToExist(column='likes'))
suite.add_expectation(gx.expectations.ExpectColumnToExist(column='comments'))
suite.add_expectation(gx.expectations.ExpectColumnToExist(column='likes_per_100_views'))
suite.add_expectation(gx.expectations.ExpectColumnToExist(column='sentiment'))
suite.add_expectation(gx.expectations.ExpectColumnToExist(column='comment'))
suite.add_expectation(gx.expectations.ExpectColumnToExist(column='clean_comment'))

# Validate data types
suite.add_expectation(gx.expectations.ExpectColumnValuesToBeOfType(
    column='video_id', type_="object"
    ))
suite.add_expectation(gx.expectations.ExpectColumnValuesToBeOfType(
    column='views', type_="int64"
    ))
suite.add_expectation(gx.expectations.ExpectColumnValuesToBeOfType(
    column='likes', type_="int64"
    ))
suite.add_expectation(gx.expectations.ExpectColumnValuesToBeOfType(
    column='comments', type_="int64"
    ))
suite.add_expectation(gx.expectations.ExpectColumnValuesToBeOfType(
    column='likes_per_100_views', type_="float"
    ))
suite.add_expectation(gx.expectations.ExpectColumnValuesToBeOfType(
    column='sentiment', type_="int64"
    ))
suite.add_expectation(gx.expectations.ExpectColumnValuesToBeOfType(
    column='comment', type_="object"
    ))
suite.add_expectation(gx.expectations.ExpectColumnValuesToBeOfType(
    column='clean_comment', type_="object"
    ))

# Validate no empty values exist
suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column='video_id'))
suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column='views'))
suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column='likes'))
suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column='comments'))
suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column='likes_per_100_views'))
suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column='sentiment'))
suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column='comment'))
suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column='clean_comment'))

# Validate results
validation_results = batch.validate(suite)
print(validation_results.success)



# Write complete data to .csv file. This will likely need to updated in the context
# of our MLOps framework—i.e. data versioning
#complete_data_path = f'sentiment_analysis_data_{datetime.today().strftime("%Y%m%d")}.parquet'
complete_data_path = 'sentiment_analysis_data.parquet'
complete_data.to_parquet(complete_data_path, index = False)