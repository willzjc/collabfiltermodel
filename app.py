# importing libraries
import pandas as pd
import numpy as np
from numpy import ndarray
from pandas import DataFrame
from sklearn.metrics.pairwise import pairwise_distances

# pass in column names for each CSV as the column name is not given in the file and read them using pandas.

# You can check the column names from the readme file

# # reading users file:
# user_columns = ["user_id", "age", "sex", "occupation", "zip_code"]
# users: DataFrame = pd.read_csv("data/ml-100k/u.user", sep="|", names=user_columns, encoding="latin-1")

# reading ratings file:
rating_columns = ["user_id", "movie_id", "rating", "unix_timestamp"]
ratings: DataFrame = pd.read_csv("data/ml-100k/u.data", sep="\t", names=rating_columns, encoding="latin-1")

# # reading items file:
# item_columns = ["movie id", "movie title", "release date", "video release date", "IMDb URL", "unknown", "Action",
#                 "Adventure",
#                 "Animation", "Children\"s", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
#                 "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]
# items: DataFrame = pd.read_csv("data/ml-100k/u.item", sep="|", names=item_columns,
#                                encoding="latin-1")

# # After loading the dataset, we should look at the content of each file (users, ratings, items).
#
# # Looking at the user file
# print("\nUser Data :")
# print("shape : ", users.shape)
# print(users.head())

# We have 943 users in the dataset and each user has 5 features, i.e. user_ID, age, sex, occupation and zip_code. Now let’s look at the ratings file.

# Ratings Data
print("\nRatings Data :")
print("shape : ", ratings.shape)
print(ratings.head())

# We have 100k ratings for different user and movie combinations. Now finally examine the items file.

# # Item Data
# print("\nItem Data :")
# print("shape : ", items.shape)
# print(items.head())

# ### Showing Ratings
#
# rating_columns = ["user_id", "movie_id", "rating", "unix_timestamp"]
# ratings_train: DataFrame = pd.read_csv("data/ml-100k/ua.base", sep="\t", names=rating_columns, encoding="latin-1")
# ratings_test: DataFrame = pd.read_csv("data/ml-100k/ua.test", sep="\t", names=rating_columns, encoding="latin-1")
#
# print(ratings_train.shape)  # (90570, 4)
# print(ratings_test.shape)   # (9430, 4)

# Building collaborative filtering model from scratch:
# We will recommend movies based on user-user similarity and item-item similarity.
# For that, first we need to calculate the number of unique users and movies.

number_of_users = ratings.user_id.unique().shape[0]
number_of_movies = ratings.movie_id.unique().shape[0]

# Now, we will create a user-item matrix which can be used to calculate the similarity between users and items.

# Ratings looks like -> each user_id refers to a user row index
# similarly, each movie_id refers to a movie row index
#    user_id  movie_id  rating  unix_timestamp
# 0      196       242       3       881250949
# 1      186       302       3       891717742
# 2       22       377       1       878887116
# 3      244        51       2       880606923
# 4      166       346       1       886397596

# line[1] is the userId and we are subtracting 1 from it since array indexing starts from 0 = row
# line[2]-1 is the movie id = column
# now at that specific row and column i.e, user and movie we will add line[3] which is the movie rating
data_matrix: ndarray = np.zeros((number_of_users, number_of_movies))
for line in ratings.itertuples():
    data_matrix[line[1] - 1, line[2] - 1] = line[3]


# Now, when we have rating os all the movies given by each user in a matrix we will calculate the
# similarity. We can use the pairwise_distance function from sklearn to calculate the cosine similarity.
user_similarity = pairwise_distances(data_matrix, metric='cosine')
movie_similarity = pairwise_distances(data_matrix.T, metric='cosine')

# This gives us the item-item and user-user similarity in an array form. The next step is to make
# predictions based on these similarities. Let’s define a function to do just that.

def predict(ratings: ndarray, similarity: ndarray, type: str = 'user') -> ndarray:
    pred = None
    # we are taking mean of axis=1 / rows
    # where axis0 -> y-axis and axis1 -> x-axis
    # i.e we are taking means of each row
    if type == 'user':
        mean_user_rating = ratings.mean(axis=1)
        # We use np.newaxis so that mean_user_rating has same format as ratings
        ratings_diff = (ratings - mean_user_rating[:, np.newaxis])
        pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array(
            [np.abs(similarity).sum(axis=1)]).T
    elif type == 'item':
        pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])
    return pred


user_prediction = predict(data_matrix, user_similarity, type='user')
item_prediction = predict(data_matrix, movie_similarity, type='item')

print(f"User prediction = {user_prediction}")
print(f"Item prediction = {item_prediction}")
