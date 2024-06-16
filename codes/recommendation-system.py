import numpy as np
import pandas as pd
import random
from sklearn.neighbors import NearestNeighbors

### Sample user-item interaction matrix (rows represent users, columns represent items)
##user_item_matrix = np.array([
##    [1, 0, 1, 1, 0],
##    [0, 1, 1, 1, 0],
##    [1, 1, 0, 1, 1],
##    [0, 1, 0, 0, 1],
##    [1, 1, 1, 0, 1]
##])

# Sample user IDs
user_ids = ['Alice', 'Balian', 'Centomaru', 'Drake', 'Esappu', 'Fascotoli', 'Gianluigi', 'Hazelbank', 'Ivanovic', 'Jovanca', 'Kurt', 'Leonardo', 'Manaya', 'Nedved']

# Sample item IDs
item_ids = ['Laptop', 'Chair', 'Phonecase', 'Knife', 'Photo Frame', 'Chocolate', 'DVD Player', 'SSD External', 'Shoes', 'Educational toys']

# Generate the user-item interaction matrix (rows represent users, columns represent items)
user_item = list()
for i in range(len(user_ids)):
    user_preferences = list()
    for j in range(len(item_ids)):
        pref_state = random.randint(0, 14) # random 0 to 14
        if pref_state <= 3:
            user_preferences.append(1) # 1 = to buy
        else:
            user_preferences.append(0) # 0 = not to buy
    user_item.append(user_preferences)
user_item_matrix = np.array(user_item)
print(f'user_item_matrix: \n{user_item_matrix}')
for i in range(len(user_item_matrix)):
    item_set = user_item_matrix[i]
    item_names = list()
    for j in range(len(item_set)):
        if item_set[j] == 1:
            item_names.append(item_ids[j])
    print(f'User: {user_ids[i]}')
    print(item_names)

# Define the number of neighbors for k-NN
k = random.randint(3, 5)
print(f'k = {k}')

# Create a k-NN model #kulik ini#
knn_model = NearestNeighbors(n_neighbors=k, metric='cosine', algorithm='brute')
knn_model.fit(user_item_matrix)

# Function to get k nearest neighbors for a given user #kulik ini#
def get_k_nearest_neighbors(user_index):
    distances, indices = knn_model.kneighbors([user_item_matrix[user_index]], n_neighbors=k+1)
    return distances[0][1:], indices[0][1:]  # Exclude the user itself

# Function to recommend items for a given user #kulik ini#
def recommend_items(user_index):
    distances, indices = get_k_nearest_neighbors(user_index)
    recommended_items = set()
    for index in indices:
        for item_index, item_rating in enumerate(user_item_matrix[index]):
            if user_item_matrix[user_index][item_index] == 0 and item_rating == 1:  # Item not interacted by user, but rated by neighbor
                recommended_items.add(item_ids[item_index])
    return recommended_items

# Example: Recommend items for User
for i in range(len(user_ids)):
    user_index = i
    recommended_items = recommend_items(user_index)
    print(f"Recommended items for {user_ids[user_index]}: {recommended_items}")