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

### Finally, we predict based on row user to user, and user to movie

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

### Result

User to user simularity prediction
```python
User prediction = [[ 2.06532606  0.73430275  0.62992381 ...  0.39359041  0.39304874
   0.3927712 ]
 [ 1.76308836  0.38404019  0.19617889 ... -0.08837789 -0.0869183
  -0.08671183]
 [ 1.79590398  0.32904733  0.15882885 ... -0.13699223 -0.13496852
  -0.13476488]
 ...
 [ 1.59151513  0.27526889  0.10219534 ... -0.16735162 -0.16657451
  -0.16641377]
 [ 1.81036267  0.40479877  0.27545013 ... -0.00907358 -0.00846587
  -0.00804858]
 [ 1.8384313   0.47964837  0.38496292 ...  0.14686675  0.14629808
   0.14641455]]
```

User to movie simularity  prediction (Each row is a movie, each column is a user)
```python
Item prediction = [[0.44627765 0.475473   0.50593755 ... 0.58815455 0.5731069  0.56669645]
 [0.10854432 0.13295661 0.12558851 ... 0.13445801 0.13657587 0.13711081]
 [0.08568497 0.09169006 0.08764343 ... 0.08465892 0.08976784 0.09084451]
 ...
 [0.03230047 0.0450241  0.04292449 ... 0.05302764 0.0519099  0.05228033]
 [0.15777917 0.17409459 0.18900003 ... 0.19979296 0.19739388 0.20003117]
 [0.24767207 0.24489212 0.28263031 ... 0.34410424 0.33051406 0.33102478]]

```