# %%
import gzip
from collections import defaultdict
import math
import scipy.optimize
from sklearn import svm
import numpy as np
import string
import random
import string
from sklearn import linear_model
import heapq
from sklearn.impute import SimpleImputer
import pandas as pd
import json
from sklearn.metrics import r2_score

# %%
def readGz(path):
    for l in gzip.open(path, 'rt'):
        yield json.loads(l)

# %%
# Read data
dataset = []
for l in readGz("renttherunway_final_data.json.gz"):
    dataset.append(l)
random.shuffle(dataset)

allRatings = []
for datum in dataset:
    if datum['rating'] is None:
        continue
    allRatings.append((datum['user_id'], datum['item_id'], int(datum['rating'])))

# %%
len(allRatings)

# %%
# Split train, valid, test
train_ratio = 0.6
val_ratio = 0.2
test_ratio = 0.2

train_size = int(len(dataset) * train_ratio)
val_size = int(len(dataset) * val_ratio)

ratingsTrain = allRatings[:train_size]
ratingsVal = allRatings[train_size:train_size + val_size]
ratingsTest = allRatings[train_size + val_size:]

# %%
def MSE(predictions, y):
    diff = [(pred - true)**2 for (pred, true) in zip(predictions, y)]
    return sum(diff) / len(diff)

# %%
usersPerItem = defaultdict(set) # Maps an item to the users who rated it
itemsPerUser = defaultdict(set) # Maps a user to the items that they rated
reviewsPerUser = defaultdict(list)
reviewsPerItem = defaultdict(list)
ratingDict = {} # To retrieve a rating for a specific user/item pair

for d in ratingsTrain:
    user,item = d[0], d[1]
    usersPerItem[item].add(user)
    itemsPerUser[user].add(item)
    reviewsPerUser[user].append(d)
    reviewsPerItem[item].append(d)
    ratingDict[(user,item)] = d[2]

# %%
userAverages = {}
itemAverages = {}
ratingMean = []

for u in itemsPerUser:
    rs = [ratingDict[(u,i)] for i in itemsPerUser[u]]
    userAverages[u] = sum(rs) / len(rs)
    
for i in usersPerItem:
    rs = [ratingDict[(u,i)] for u in usersPerItem[i]]
    itemAverages[i] = sum(rs) / len(rs)
    
for d in ratingsTrain:
    ratingMean.append(d[2])
    
ratingMean = sum(ratingMean) / len(ratingMean)

# %%
jaccard_users = {}
jaccard_items = {}

# %%
def Jaccard(s1, s2):
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    if denom > 0:
        return numer/denom
    return 0

# %%
def jaccardUsers(u1, u2):
    if (u1, u2) not in jaccard_users:
        sim = Jaccard(itemsPerUser[u1], itemsPerUser[u2])
        jaccard_users[(u1, u2)], jaccard_users[(u2, u1)] = sim, sim
    return jaccard_users[(u1, u2)]

def jaccardItems(i1, i2):
    if (i1, i2) not in jaccard_items:
        sim = Jaccard(usersPerItem[i1], usersPerItem[i2])
        jaccard_items[(i1, i2)], jaccard_items[(i2, i1)] = sim, sim
    return jaccard_items[(i1, i2)]

# %%
# Method 1: Collaborative Filtering based on Jaccard similarity of Items
def predictRatingByItemSimilarity(user,item):
    ratings = []
    similarities = []
    for d in reviewsPerUser[user]:
        i2 = d[1]
        if i2 == item: continue
        ratings.append(d[2] - itemAverages[i2])
        similarities.append(jaccardItems(item, i2))
    if (sum(similarities) > 0):
        weightedRatings = [(x*y) for x,y in zip(ratings,similarities)]
        return itemAverages[item] + sum(weightedRatings) / sum(similarities)
    else:
        # User hasn't rated any similar items
        return ratingMean

# %%
# Method 2: Collaborative Filtering based on Jaccard similarity of Users
def predictRatingByUserSimilarity(user,item):
    ratings = []
    similarities = []
    for d in reviewsPerItem[item]:
        u2 = d[0]
        if u2 == user: continue
        ratings.append(d[2] - userAverages[u2])
        similarities.append(jaccardUsers(user, u2))
    if (sum(similarities) > 0):
        weightedRatings = [(x*y) for x,y in zip(ratings,similarities)]
        return itemAverages[item] + sum(weightedRatings) / sum(similarities)
    else:
        # User hasn't rated any similar items
        return ratingMean

# %%
y = [d[2] for d in ratingsTest]
predictions = []
for u, i, r in ratingsTest:
    predictions.append(predictRatingByItemSimilarity(u, i))
mse1 = MSE(y, predictions)
r2_1 = r2_score(y, predictions)

predictions = []
for u, i, r in ratingsTest:
    predictions.append(predictRatingByUserSimilarity(u, i))
mse2 = MSE(y, predictions)
r2_2 = r2_score(y, predictions)

# %%
mse1, mse2, r2_1, r2_2


