import gzip
from collections import defaultdict
import math
import scipy.optimize
from sklearn import svm
import numpy
import string
import random
import string
from sklearn import linear_model
import tqdm

def readCSV(path):
    f = gzip.open(path, 'rt')
    f.readline()
    for l in f:
        u,b,r = l.strip().split(',')
        r = int(r)
        yield u,b,r

allRatings = []
for l in readCSV("train_Interactions.csv.gz"):
    allRatings.append(l)

ratingsTrain = allRatings[:190000]
ratingsValid = allRatings[190000:]
ratingsPerUser = defaultdict(list)
ratingsPerItem = defaultdict(list)
BooksPerUser= defaultdict(set)
UsersPerBook= defaultdict(set)
books = set()
users = set()
for u,b,r in ratingsTrain:
    ratingsPerUser[u].append((b,r))
    ratingsPerItem[b].append((u,r))
    UsersPerBook[b].add(u)
    BooksPerUser[u].add(b)
    books.add(b)
    users.add(u)

"""# Task 1"""

# Copied from baseline code
bookCount = defaultdict(int)
totalRead = 0
for user,book,_ in readCSV("train_Interactions.csv.gz"):
    bookCount[book] += 1
    totalRead += 1 #total interactions

mostPopular = [(bookCount[x], x) for x in bookCount] #bookcount[book],book
mostPopular.sort()
mostPopular.reverse()

books = set(bookCount.keys())
NewRatingsValid = []
for u,b,r in ratingsValid:
  not_read_list = books - BooksPerUser[u]
  unread_book = random.choice(list(not_read_list))
  NewRatingsValid.append((u,unread_book,0))

for u,b,r in ratingsValid:
  NewRatingsValid.append((u,b,1))

target_labels= [d[2] for d in NewRatingsValid]

def accuracy(labels,predictions):
  correct = 0
  for l, p  in zip(labels,predictions):
    if l==p:
      correct += 1
  return correct/len(predictions)

"""Finding appropriate threshold based on Baseline Model (most popular model)"""

threshold = 0
acc2 = 0
for thres in tqdm.tqdm(range(50,101)):
  return2 = set() #set of books
  count = 0
  for ic, i in mostPopular:
    count += ic
    return2.add(i)
    if count > totalRead * thres * 0.01: break
  predictions = []
  for u, b,_ in NewRatingsValid:
    if b in return2:
      predictions.append(1)
    else:
      predictions.append(0)
  acc_new = accuracy(target_labels,predictions)
  if acc2 < acc_new:
    acc2 = acc_new
    threshold = thres * 0.01

"""Finding appropriate threshold based on Jaccard similarity."""

threshold

#threshold = 0.73

return3 = set() #set of books
count = 0
for ic, i in mostPopular:
  count += ic
  return3.add(i)
  if count > totalRead * 0.73: break

# Commented out IPython magic to ensure Python compatibility.
# %pip install cornac

import pandas as pd

# File path to train_Interactions.csv
train_file = "train_Interactions.csv"
test_file = "pairs_Read.csv"
# Read the CSV file and skip the header
df_train = pd.read_csv(train_file, skiprows=1, names=["userID", "bookID", "rating"])

# Display the first few rows of the dataset
print(df_train.head())

test_df = pd.read_csv(test_file, skiprows=1, names=["userID", "bookID", "predicition"])
print(test_df.head())

import cornac
from sklearn.model_selection import train_test_split
train_set = cornac.data.Dataset.from_uir(df_train.itertuples(index=False), seed=42)
bpr = cornac.models.BPR(
    k=16,
    max_iter=100,
    learning_rate=0.01,
    lambda_reg=0.001,
    verbose=True,
    seed=42
)

bpr.fit(train_set)

inverse = {item: ind for ind, item in enumerate(bpr.item_ids)}
inverse_users = {item: ind for ind, item in enumerate(bpr.user_ids)}

grouped = test_df.groupby("userID").agg({
    "bookID": list,
}).reset_index()

predictions = open("predictions_Read.csv", 'w')
for l in open("pairs_Read.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        break
for index, rows in grouped.iterrows():
    inds = [inverse[book] if book in books else 'b33877827' for book in rows["bookID"]]
    try:
      ranks = bpr.rank(inverse_users[rows['userID']], inds)
    except:
      for book in rows["bookID"]:
        if book in return3:
          predictions.write(rows['userID'] + ',' + book +',' + "1\n")
        else:
          predictions.write(rows['userID'] + ',' + book +','+ "0\n")

    order = ranks[0]

    for i, r in enumerate(order):
      if i+1<=len(order)/2:
        predictions.write(rows['userID'] + ',' + bpr.item_ids[r] +',' + "1\n")
      else:
        predictions.write(rows['userID'] + ',' + bpr.item_ids[r] +','+ "0\n")
predictions.close()

"""# Task 2"""

# Commented out IPython magic to ensure Python compatibility.
# %pip install implicit
# %pip install surprise

import gzip
import random
import scipy
import tensorflow as tf
from collections import defaultdict
from surprise import SVD, Reader, Dataset
from surprise.model_selection import train_test_split,GridSearchCV
from surprise import accuracy

reader = Reader(line_format='user item rating', sep=',',skip_lines=1)
data = Dataset.load_from_file("train_Interactions.csv", reader=reader)

model = SVD()

trainset, validset = train_test_split(data, test_size=.05)

param_grid = {
    'n_factors': [1,5,10],
    'lr_all': [0.01],
    'reg_all': [ 0.1,0.2,0.3]
}

gs = GridSearchCV(SVD, param_grid, measures=['mse'], cv=3, n_jobs=-1)
gs.fit(data)

print(f"Best MSE: {gs.best_score['mse']}")
print(f"Best Parameters: {gs.best_params['mse']}")

best_model = gs.best_estimator['mse']
best_model.fit(trainset)

predictions = best_model.test(validset)

mse = accuracy.mse(predictions)
print(f"Validation MSE: {mse}")

import pandas as pd
df = pd.read_csv("pairs_Rating.csv")
df['prediction'] = 0
df = list(df.to_records(index=False))
predictions = best_model.test(df)

filep = open("predictions_Rating.csv", 'w')
for l in open("pairs_Rating.csv"):
    if l.startswith("userID"): # header
        filep.write(l)
        continue
for index in range(len(predictions)):
  predictions[index].est
  filep.write(predictions[index].uid + ',' + predictions[index].iid + ',' + str(predictions[index].est) + '\n')

filep.close()