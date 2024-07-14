# Objective: Build a movie recommendation system using collaborative filtering

# Data Source: MovieLens dataset

# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import TruncatedSVD

# Import Data
ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('movies.csv')

# Describe Data
print("Ratings Data:")
print(ratings.head())
print(ratings.describe())

print("\nMovies Data:")
print(movies.head())
print(movies.describe())

# Data Visualization
plt.figure(figsize=(8, 6))
sns.countplot(ratings['rating'])
plt.title('Distribution of Ratings')
plt.xlabel('Rating')
plt.ylabel('Count')
plt.show()

ratings_per_movie = ratings.groupby('movieId').size()
plt.figure(figsize=(8, 6))
plt.hist(ratings_per_movie, bins=50)
plt.title('Number of Ratings per Movie')
plt.xlabel('Number of Ratings')
plt.ylabel('Count')
plt.show()

# Data Preprocessing
data = pd.merge(ratings, movies, on='movieId')
pivot_table = data.pivot_table(index='userId', columns='title', values='rating')
pivot_table = pivot_table.fillna(0)

# Define Target Variable (y) and Features Variables (X)
X = pivot_table.values

# Train Test Split
train_data, test_data = train_test_split(X, test_size=0.2, random_state=42)

# Modeling
svd = TruncatedSVD(n_components=50, random_state=42)
svd.fit(train_data)
train_svd = svd.transform(train_data)
test_svd = svd.transform(test_data)

# Model Evaluation
train_reconstructed = np.dot(train_svd, svd.components_)
test_reconstructed = np.dot(test_svd, svd.components_)

train_mse = mean_squared_error(train_data, train_reconstructed)
test_mse = mean_squared_error(test_data, test_reconstructed)

print(f'Train MSE: {train_mse}')
print(f'Test MSE: {test_mse}')


# Prediction
def predict_ratings(user_id, model, pivot_table):
    user_idx = pivot_table.index.get_loc(user_id)
    user_ratings = model[user_idx, :]
    return pd.Series(user_ratings, index=pivot_table.columns)


user_id = 1
predicted_ratings = predict_ratings(user_id, test_reconstructed, pivot_table)
print("Top 10 movie recommendations for User 1:")
print(predicted_ratings.sort_values(ascending=False).head(10))
