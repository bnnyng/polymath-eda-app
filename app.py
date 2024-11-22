import streamlit as st
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt

# st.session_state.comment_path = "comments_cleaned.json"
# st.session_state.ratings_path = "comment_ratings_all_projects.json"

with open("comment_ratings_all_projects.json", "r") as f:
    ratings_data = pd.DataFrame(json.load(f))
with open("comments_cleaned.json", "r") as f:
    comment_data = pd.DataFrame(json.load(f))
project_ids = list(comment_data["project-id"].unique())

project_data = {}
for project_id in project_ids:
    project_data[project_id] = ratings_data[
        ratings_data["id"].apply(lambda x: x.split('-')[1] == project_id)
    ]

st.header("View data")
st.subheader("All comments")
st.dataframe(comment_data)
st.subheader("All ratings")
st.dataframe(ratings_data)

st.header("Plots")
threshold = st.number_input(
    label="Enter a threshold value between 0 and 1:",
    min_value = float(0),
    max_value = float(1),
    value = 0.7
)

for i, (id, ratings) in enumerate(project_data.items()):
    ratings_num = ratings.iloc[:, 2:].apply(pd.to_numeric, errors='coerce')
    ratings_num = (ratings_num > threshold).astype(int) 
    ratings_num.columns = ratings_num.columns.str.replace("category-", "", regex=False)
    ratings_sums = ratings_num.sum()
    
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.bar(ratings_sums.index, ratings_sums.values)
    ax.set_title(f"{id} GPT Ratings With Threshold {threshold}")
    ax.set_ylabel(f"Number of comments with rating above {threshold}")
    ax.set_xlabel("Categories")
    st.pyplot(fig)