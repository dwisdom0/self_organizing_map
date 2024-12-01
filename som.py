import plotly.graph_objects as go

from copy import deepcopy
from math import sqrt
from pprint import pprint
from random import random

from tqdm import tqdm

# algorithm

# 2d grid of vectors
# get an input sample (one vector)
# find the vector that's closest in the grid. that's the "best matching unit" BMU
# update that BMU and some of its neighbors in the grid


# usually there's a learning schedule 
# so learning rate and "closeness" are usually parameterized by iteration step


class SOM:
  def __init__(self, grid_size, vec_size):
    self.__grid_size = grid_size
    self.__vec_size = vec_size
    self.__grid = [[self.__rand_vec(vec_size) for _ in range(grid_size)] for _ in range(grid_size)]

  def __rand_vec(self, size):
    return [random() for _ in range(size)]

  def __repr__(self):
    s = ''
    for row in self.__grid:
      for vec in row:
        s += f'{vec[0]:.2f}...{vec[-1]:.2f}  '
      s += '\n'
    return s

  def __str__(self):
    return self.__repr__()

  def __find_bmu(self, in_vec):
    # TODO: using euclidean distance so I have to pick a huge number
    # If i could use cosine distance this could be 1
    # maybe something to look into later

    # could also initialize it using the first point on the grid I guess
    min_dist = 1e30
    min_i = None
    min_j = None
    for i, row in enumerate(self.__grid):
      for j, vec in enumerate(row):
        dist = sqrt(sum([(a - b)**2 for a, b in zip(vec, in_vec)]))
        if dist < min_dist:
          min_dist = dist
          min_i = i
          min_j = j


    if min_i is None or min_j is None:
      raise ValueError(f'no grid vectors within {min_dist} of {in_vec}\n\n{self.__repr__()}')
    return min_i, min_j


  @staticmethod
  def taxicab(v1, v2):
    return sum([abs(a - b) for a, b in zip(v1, v2)])


  def training_step(self, in_vec, learning_rate):
    """
    update function:
    W_i = W_i + "closeness to BMU in grid" * learning_rate * (input - W_i)
    """
    bmu_row, bmu_col = self.__find_bmu(in_vec)
    for i, row in enumerate(self.__grid):
      for j, vec in enumerate(row):
        # bit of a hack since I'm using 1/taxicab for dist
        # can't do 1/0 for the BMU node
        if i == bmu_row and j == bmu_col:
          grid_mult = 1
        else:
          # just going to use manhattan distance for the grid distance discount
          grid_mult = 1 / self.taxicab((i, j), (bmu_row, bmu_col))
        new_vec = [old + grid_mult * learning_rate * (in_val - old)  for old, in_val in zip(vec, in_vec)]
        self.__grid[i][j] = deepcopy(new_vec)


  def train(self, data_points, epochs=1):
    learning_rate = 2.0
    for epoch in tqdm(range(epochs), ascii=True):
      print(f'{learning_rate=}')
      for data_point in tqdm(data_points, ascii=True):
        self.training_step(data_point, learning_rate)
      learning_rate *= 0.90

  def get_grid_nodes(self):
    grid_nodes = []
    for row in self.__grid:
      for node in row:
        grid_nodes.append(node)
    return grid_nodes

  def get_grid_size(self):
    return self.__grid_size

  def classify(self, in_vec):
    return self.__find_bmu(in_vec)


def gen_clustered_vectors(num_vectors, dim):
  vecs = []
  for _ in range(1000):
    new_record = []
    if random() < 0.6:
      for _ in range(dim):
        new_record.append(random())
    else:
      for _ in range(dim):
        new_record.append(random() + 3)
    vecs.append(new_record)

  return vecs


def main():

  dim = 20
  data = gen_clustered_vectors(1000, dim)

  pprint(data[:5])

  print('\n\n')

  som_grid_size = 20
  som = SOM(grid_size=som_grid_size, vec_size=dim)
  print(som)

  print('\n\n')

  som.train(data, epochs=3)
  print(som)

  f = go.Figure(go.Scatter(
    x=[n[0] for n in som.get_grid_nodes()],
    y=[n[1] for n in som.get_grid_nodes()],
    name='SOM grid',
    mode='markers+lines',
  ))

  f.add_trace(go.Scatter(
    x=[d[0] for d in data],
    y=[d[1] for d in data],
    name='data',
    mode='markers',
  ))

  f.show()

  # use the SOM to visualize our high-dimensional data in 2D
  # by building up a heatmap of where our points land on the SOM grid
  preds = []
  for _ in range(som_grid_size):
    preds.append([0 for _ in range(som_grid_size)])

  for d in data:
    i, j = som.classify(d)
    preds[i][j] += 1

  f2 = go.Figure(go.Heatmap(
    z=preds,
    #colorscale='Hot_r',
    #colorscale='Purples',
    colorscale='Mint',
  ))
  f2.show()






if __name__ == '__main__':
  main()

