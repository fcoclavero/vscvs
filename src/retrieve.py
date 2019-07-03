import torch

from datetime import datetime
from sklearn.neighbors import NearestNeighbors

k = 16

start = datetime.now()

with torch.no_grad():
    results  = NearestNeighbors(n_neighbors=k, algorithm='brute', metric='cosine').fit(torch.stack(vectors).squeeze())

distances, indices = results.kneighbors(vectors[0].reshape(1,-1))

print('KNN search duration: %s' % (datetime.now() - start))