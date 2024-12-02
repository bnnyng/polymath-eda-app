import streamlit as st
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# st.session_state.comment_path = "comments_cleaned.json"
# st.session_state.ratings_path = "comment_ratings_all_projects.json"

st.session_state.ratings_files = [
    "comment_ratings_all_projects_1.json",
    "comment_ratings_all_projects_2.json"
]
st.session_state.ratings_data = []
for file_name in st.session_state.ratings_files:
    with open(file_name, "r") as f:
        df = pd.DataFrame(json.load(f))
        st.session_state.ratings_data.append(df)
with open("comments_cleaned.json", "r") as f:
    st.session_state.comment_data = pd.DataFrame(json.load(f))
with open("cognitive_category_definitions.json", "r") as f:
    st.session_state.category_defs = json.load(f)
st.session_state.categories = list(st.session_state.category_defs.keys())
# project_ids = list(st.session_state.comment_data["project-id"].unique())

# project_data = {}
# for project_id in project_ids:
#     project_data[project_id] = ratings_data[
#         ratings_data["id"].apply(lambda x: x.split('-')[1] == project_id)
#     ]

#####   VIEW RAW DATA   #####

st.header("Data summary")
st.subheader("All comments")
st.dataframe(st.session_state.comment_data)
for i in range(len(st.session_state.ratings_data)):
    st.subheader(f"GPT ratings - Trial {i+1}")
    st.dataframe(st.session_state.ratings_data[i])

def compute_category_correlations(df1, df2):
    return df1.iloc[:, 2:].corrwith(df2.iloc[:, 2:])

st.subheader("GPT-GPT correlations")
correlations_data = compute_category_correlations(st.session_state.ratings_data[0], st.session_state.ratings_data[1])
st.dataframe(correlations_data)

#####   PCA   #####

@st.fragment
def pca(ratings_data, n_components, cats=st.session_state.categories):
    data = ratings_data[[f"category-{category}" for category in cats]]
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(data)
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    st.write(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    st.write(f"Cumulative variance: {cumulative_variance}")

    pca_comments = pd.DataFrame(principal_components, columns=[f"PC{i+1}" for i in range(n_components)])
    pca_categories = pd.DataFrame(
        data=pca.components_,
        columns=list(ratings_data.columns)[2:],
        index=[f"PC{i+1}" for i in range(pca.n_components_)]
    )

    # DATA
    # with col2:
    st.write("Principal component values per category:")
    st.dataframe(pca_categories, height=200)
    st.write("Principal component values per comment:")
    st.dataframe(pca_comments, height=200)
    

    def compute_pc1_correlations(pca_comments, pca_categories, ratings_data):
        pc1_data = pca_comments["PC1"]
        category_data = ratings_data[[f"category-{category}" for category in st.session_state.categories]]
        return category_data.corrwith(pc1_data)
    st.write("Correlation between category ratings and PC1 across all comments:")
    st.dataframe(compute_pc1_correlations(pca_comments, pca_categories, ratings_data))

    return pca_comments, pca_categories


st.header("PCA")
# INPUTS
# col1, col2 = st.columns(2)
# with col1:
ratings_idx = st.selectbox(
    label="Select GPT trial to view:",
    options = [1, 2]
)
ratings_data = st.session_state.ratings_data[ratings_idx-1]
n_components = st.number_input(
    label="Set the number of components:",
    min_value=2,
    max_value=14,
    value=2
)
pca_comments, pca_categories = pca(ratings_data, n_components)


@st.fragment
def plot_category_pcs():
    category_subset = st.multiselect(
        label="Select categories:",
        options=st.session_state.categories,
        default=None
    )
    if category_subset == None:
        category_subset = st.session_state.categories
    pca_comments_sub, pca_categories_sub = pca(ratings_data, n_components=2, cats=category_subset)

    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots()
        ax.scatter(pca_comments_sub["PC1"], pca_comments_sub["PC2"], alpha=0.3)
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.set_title("Principal component values for comments")
        st.pyplot(fig)
    with col2:
        fig, ax = plt.subplots()
        ax.scatter(pca_categories_sub["PC1"], pca_categories_sub["PC2"], alpha=0.3)
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.set_title("Principal component values for categories")
        st.pyplot(fig)

@st.fragment
def get_mostly_low_comments():
    col1, col2 = st.columns(2)
    with col1:
        threshold = st.number_input(
            label="Enter maximum absolute PC value:",
            min_value = float(0),
            value = 0.0
        )
        n_pcs = st.number_input(
            label="Enter number of PCs to consider:",
            min_value = len(pca_comments.columns)
        )
    with col2:
        low_comment_data = (pca_comments.copy().abs() <= threshold).sum(axis=1)
        low_comment_idx = low_comment_data[low_comment_data >= n_pcs].index.to_list()
        low_comments = st.session_state.comment_data.loc[low_comment_idx]
        low_comments = pd.concat([low_comments[["id", "content"]], pca_comments.loc[low_comment_idx]], axis=1)
        st.dataframe(low_comments, height=200)


# st.subheader("Plot PCs with category subset")
# plot_category_pcs()
st.subheader("Comments with low PCs")
get_mostly_low_comments()






#low_documents = pca_df[pca_df["low_pc_count"] >= min_low_pcs]
#         # Count the number of PCs below the threshold for each document
# pca_df["low_pc_count"] = (pca_df.drop("document_id", axis=1) < threshold).sum(axis=1)

# # Calculate the number of PCs needed to meet the "most PCs" condition
# min_low_pcs = int(proportion * (pca.n_components_))

# # Filter documents that are low in "most PCs"
# low_documents = pca_df[pca_df["low_pc_count"] >= min_low_pcs]

# # Display results
# print("Documents low in most PCs:")
# print(low_documents)


# # Find the document with the lowest value for each component
# lowest_docs = {}
# for pc in pca_df.columns[:-1]:  # Skip the 'document_id' column
#     lowest_docs[pc] = pca_df.nsmallest(1, pc)

# # Display results
# for pc, doc in lowest_docs.items():
#     print(f"Lowest document in {pc}:")
#     print(doc)





    # fig, ax = plt.subplots()
    # # Plot 
    # ax.scatter(pca_comments["PC1"], pca_comments["PC2"], alpha=0.3)
    # ax.set_xlabel("Principal Component 1")
    # ax.set_ylabel("Principal Component 2")
    # ax.set_title("Principal component values for ")
    # st.pyplot(fig)
    # st.dataframe(pca_comments)



# @st.fragment
# def plot_histograms():
#     threshold = st.number_input(
#         label="Enter a threshold value between 0 and 1:",
#         min_value = float(0),
#         max_value = float(1),
#         value = 0.7
#     )

#     for i, (id, ratings) in enumerate(project_data.items()):
#         ratings_num = ratings.iloc[:, 2:].apply(pd.to_numeric, errors='coerce')
#         ratings_num = (ratings_num > threshold).astype(int) 
#         ratings_num.columns = ratings_num.columns.str.replace("category-", "", regex=False)
#         ratings_sums = ratings_num.sum()
        
#         fig, ax = plt.subplots(figsize=(15, 5))
#         ax.bar(ratings_sums.index, ratings_sums.values)
#         ax.set_title(f"{id} GPT Ratings With Threshold {threshold}")
#         ax.set_ylabel(f"Number of comments with rating above {threshold}")
#         ax.set_xlabel("Categories")
#         st.pyplot(fig)



# st.header("PCA")
# pca()

# st.header("Plots")
# with st.container(height=500):
#     plot_histograms()

# # Select numerical columns
# data = df_ratings.iloc[:, 2:]

# # Get components
# n_components=2
# pca = PCA(n_components=n_components)
# principal_components = pca.fit_transform(data)
# df_pca = pd.DataFrame(principal_components, columns=[f"PC{i+1}" for i in range(n_components)])

# cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

# print("Explained variance ratio:", pca.explained_variance_ratio_)
# print("Cumulative variance:", cumulative_variance)