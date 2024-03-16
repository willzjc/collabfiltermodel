# Notes

This is a collab working model with User and Movies

Basically based on a user's ratings of certain movies, what other movies is the user likely to like?

There are multiple objects, but only the Ratings dataframe matters

### Ratings structure

It looks like

```
# Ratings looks like -> each user_id refers to a user row index
# similarly, each movie_id refers to a movie row index
#    user_id  movie_id  rating  unix_timestamp
# 0      196       242       3       881250949
# 1      186       302       3       891717742
# 2       22       377       1       878887116
# 3      244        51       2       880606923
# 4      166       346       1       886397596

```

### Transformation to matrix

We then use the following to transform into a data matrix of `${number of users}` by `${number of movies}` size

```python
data_matrix: ndarray = np.zeros((n_users, n_items))
for line in ratings.itertuples():
    data_matrix[line[1] - 1, line[2] - 1] = line[3]
```

It will become something that looks like this

```
array([[5., 3., 4., ..., 0., 0., 0.],
       [4., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       ...,
       [5., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 5., 0., ..., 0., 0., 0.]])
```

### Pairwise distance a matrix

We pairwise distance each element of the matrix with

```python
user_similarity = pairwise_distances(data_matrix, metric='cosine')
item_similarity = pairwise_distances(data_matrix.T, metric='cosine')
```

#### But what is pair wise distance????

Imagine you have a box of toys, and you want to know how far apart each toy is from all the other toys. Pairwise distance is like a game to find that out!

Here's how it works:

1. Pick any two toys.
2. Pretend you have a super string that can stretch any distance.
3. Put one end of the string on the first toy, and stretch it all the way to the other toy.
4. See how long the string is! That's the distance between those two toys.

Pairwise distance does this for every single pair of toys in the box! It's like a big chart that shows how far each toy is from every other toy.

So, if you ever want to know which toy car is closest to your teddy bear, or how far your doll is from all your building blocks, pairwise distance can help you figure it out!

### Finally, we predict based on row (user based), or column (movie based)

```python
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
item_prediction = predict(data_matrix, item_similarity, type='item')
```