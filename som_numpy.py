import numpy as np

from functools import lru_cache

from tqdm import tqdm

# algorithm

# 2d grid of vectors
# get an input sample (one vector)
# find the vector that's closest in the grid. that's the "best matching unit" BMU
# update that BMU and some of its neighbors in the grid


# usually there's a learning schedule 
# so learning rate and "closeness" are usually parameterized by iteration step


class SOM:
  def __init__(self, grid_size, vec_size, rng_seed=None):
    self.__rng = np.random.default_rng(seed=rng_seed)
    self.__grid_size = grid_size
    self.__vec_size = vec_size
    self.__grid = self.__rng.uniform(
      low=0.0,
      high=1.0,
      size=(grid_size, grid_size, vec_size)
    )


  def __repr__(self):
    s = ''
    for row in self.__grid:
      for vec in row:
        s += f'{vec[0]:.2f}...{vec[-1]:.2f}  '
      s += '\n'
    return s


  def __find_bmu(self, in_vec):
    # calculate the educlidean distance between in_vec and each node of the grid
    # then return the indices of the grid node whose vector is closest to in_vec

    # This is the current perf bottleneck
    # could prune the grid with a kd tree
    # but I don't want to bother doing that
    dists = np.apply_along_axis(
      func1d=self.euclidean_dist,
      arr=self.__grid,
      axis=2,
      # kwargs to pass to euclidean_dist()
      v2=in_vec
    )
    return np.unravel_index(np.argmin(dists), dists.shape)


  @staticmethod
  def euclidean_dist(v1, v2):
    return np.sqrt(np.sum((v1-v2)**2))


  @staticmethod
  @lru_cache(maxsize=1000)
  def taxicab_dist(v1, v2):
    # can't use numpy arrarys as keys
    # so can't cache this if we're calling it with numpy arrays
    # and I think it would probably be slower to convert between
    # numpy arrays and tuples
    # than it is to just do the whole calculation in python
    # I guess I can check
    # caching this made it go away from the profile
    # converting tuples to numpy arrays brought it back
    # so I'm going to stick with the cached python implementation
    return sum([abs(a-b) for a, b in zip(v1,v2)])


  @staticmethod
  @lru_cache(maxsize=1000)
  def calc_grid_dists(grid_size, bmu_row, bmu_col):
    grid_dists = []
    for r in range(grid_size):
      new_row = []
      for c in range(grid_size):
        if r == bmu_row and c == bmu_col:
          # this will make a plus of 1s
          # like
          #     212
          #     111
          #     212
          # 
          # since the center 1 there should be 0
          # but I'm taking the multiplicative inverse of these values
          # so I can't have 0
          # idk
          # It worked fine before
          # just something to think about
          # maybe make it 0.5 or something
          #
          # changing it to 0.5 actually made a huge difference
          # now it gets pulled way more into each cluster
          # and it stretches out a lot more.
          # When this was 1,
          # it would hold the grid shape more and only deform a little bit
          #
          # 0.1 is probably a little bit too far
          new_row.append(0.5)
          continue
        new_row.append(SOM.taxicab_dist(
          (r,c),
          (bmu_row, bmu_col),
        ))
      grid_dists.append(new_row)

    # wikipedia offers 4 different neighborhood functions
    # but obviously there are more so there are tons to try
    # 1. 1 if you/re close enough to BMU, 0 otherwise
    # 2. 1/2**x (halve the value every step farthar away from BMU
    # 3. "guassian" I don't know what what means
    #    I assume it's a gaussian centered on BMU
    #    and then std = 1 grid step?
    #    I don't really get how you pick a gaussian
    #    maybe you can define it in terms of learning rate or epoch
    #    earlier ones could be wider and later ones could be more focused
    # 4. Maxican hat distribution
    #    same as the gaussian
    #    the citation for those two is paywalled

    # I'll try 1/2**x
    # that's the best so far
    # what about 1/5**x
    # that might actually be better? it got twisted but more of the points
    # are in the clusters
    # 1/3**x looks similar to 1/2**x
    # 1/4**x looks similar to 1/5**x
    # kind of wierd that 1/4**x and 1/5**x always twist
    # but the lower bases don't
    # must be something about my data and learning rate
    return 1.0 / 3**np.array(grid_dists)


  def training_step(self, in_vec, learning_rate):
    """
    update function:
    W_i = W_i + "closeness to BMU in grid" * learning_rate * (input - W_i)
    """

    # how can I vectorize this
    # since I need the current i,j index for a lot of these calculations
    # make a matrix of grid_mults (that might be slow?)
    # then multiply that by the learning_rate 
    # then multiply that by in_vec - old_vec
    # then add that to old_vec
    # okay so I only need the indexes for creating the grid_mults
    # not too bad

    bmu_row, bmu_col = self.__find_bmu(in_vec)
    grid_dists = self.calc_grid_dists(self.__grid_size, bmu_row, bmu_col)
    grid_mults = grid_dists * learning_rate

    # can do elementwise multiplication if I add an extra axis
    # to match the dimension of self.__grid
    # so then I can multiply each correction vector
    # with the corresponding scalar grid_mult
    grid_mults = grid_mults.reshape(*grid_mults.shape, 1)

    updates = in_vec - self.__grid
    updates *= grid_mults
    self.__grid = self.__grid + updates


  def train(self, data_points, epochs=1):
    learning_rate = 2.0
    for epoch in tqdm(range(epochs), ascii=True):
      print(f'{learning_rate=}')
      for data_idx in tqdm(self.__rng.permutation(len(data_points)), ascii=True):
        self.training_step(data_points[data_idx], learning_rate)
      learning_rate *= 0.90

  def get_grid_nodes(self):
    grid_nodes = []
    for row in self.__grid:
      for node in row:
        grid_nodes.append(node)
    return grid_nodes

  def classify(self, in_vec):
    return self.__find_bmu(in_vec)

  def get_grid_size(self):
    return self.__grid_size

