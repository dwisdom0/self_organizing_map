import plotly.express as px
import plotly.graph_objects as go
import numpy as np

from random import random

from som_numpy import SOM


def gen_4_corners(n):
  """
  make a bunch of 2d vectors that look like
  
  ******      ******
  ******      ******
  ******      ******

  ******      ******
  ******      ******
  ******      ******
  """
  vecs = []
  for _ in range(n):
    new_vec = []
    roll = random()
    if roll < 0.1:
      # 0,0 - 1,1
      new_vec.append(random())
      new_vec.append(random())
    elif roll < 0.3:
      # 3,3 - 4,4
      new_vec.append(random() + 3)
      new_vec.append(random() + 3)
    elif roll < 0.7:
      # 0,3 - 1,4
      new_vec.append(random())
      new_vec.append(random() + 3)
    else:
      # 3,0 - 4, 1
      new_vec.append(random() + 3)
      new_vec.append(random())
    vecs.append(new_vec)
  return vecs



def gen_3d_clusters(n):
  """
  make a bunch of 3d vectors
  in three clusters
  """
  vecs = []
  for _ in range(n):
    roll = random()
    if roll < 0.3:
      vecs.append([random() + 3, random(), random()])
    elif roll < 0.5:
      vecs.append([random(), random() + 3, random()])
    else:
      vecs.append([random() + 3, random() + 3, random() + 3])
  return vecs
    


    
 
def get_adjacent_nodes(r, c, grid_size):
  if grid_size <= 1:
    raise ValueError(f"grid must be at least 2x2, got {grid_size}")
  grid_size -= 1
  adj_nodes = set()
  adj_nodes.add((max(0, r-1), c))
  adj_nodes.add((min(grid_size, r+1), c))
  adj_nodes.add((r, max(0, c-1)))
  adj_nodes.add((r, min(grid_size, c+1)))
  return adj_nodes



def plot_2d(data, som, save=False):
  f = go.Figure(go.Scatter(
    x=[d[0] for d in data],
    y=[d[1] for d in data],
    mode='markers',
    name='data',
    marker=dict(
      color='#00cc96',
    )
  ))

  som_grid_size = som.get_grid_size()
  dim = 2
  
  grid = np.array(som.get_grid_nodes()).reshape(som_grid_size, som_grid_size, dim)

  mapping = {}
  for r, row in enumerate(grid):
    for c, vec in enumerate(row):
      mapping[(r, c)] = vec[:2]

  edge_xs = []
  edge_ys = []
  for r, row in enumerate(grid):
    for c, vec in enumerate(row):
      start_pos = mapping[(r,c)]
      adj_nodes = get_adjacent_nodes(r, c, som_grid_size)
      for node in adj_nodes:
        end_pos = mapping[(node[0], node[1])]
        edge_xs.append(start_pos[0])
        edge_xs.append(end_pos[0])
        edge_xs.append(None)
        edge_ys.append(start_pos[1])
        edge_ys.append(end_pos[1])
        edge_ys.append(None)

  f.add_trace(go.Scatter(
    x=edge_xs,
    y=edge_ys,
    mode='lines',
    name='SOM edges',
    marker=dict(
      color='#636efa',
    )
  ))

  node_xs = []
  node_ys = []
  for row in grid:
    for vec in row:
      node_xs.append(vec[0])
      node_ys.append(vec[1])

  f.add_trace(go.Scatter(
    x=node_xs,
    y=node_ys,
    mode='markers',
    name='SOM nodes',
    marker=dict(
      color='#ef553b',
    )
  ))

  f.update_layout(
    title=dict(
      text="Fit a self-organizing map to the data"
    )
  )

  f.show()
  f.write_html("fit_plot.html", include_plotlyjs='cdn')


def plot_3d(data, som, save=False):
  f = go.Figure(go.Scatter3d(
    x=[d[0] for d in data],
    y=[d[1] for d in data],
    z=[d[2] for d in data],
    name='data',
    mode='markers',
    marker=dict(
      color='#00cc96',
    )
  ))

  som_grid_size = som.get_grid_size()
  dim = 3 

  grid = np.array(som.get_grid_nodes()).reshape(som_grid_size, som_grid_size, dim)

  mapping = {}
  for r, row in enumerate(grid):
    for c, vec in enumerate(row):
      mapping[(r, c)] = vec

  edge_xs = []
  edge_ys = []
  edge_zs = []
  for r, row in enumerate(grid):
    for c, vec in enumerate(row):
      start_pos = mapping[(r, c)]
      adj_nodes = get_adjacent_nodes(r, c, som_grid_size)
      for node in adj_nodes:
        end_pos = mapping[(node[0], node[1])]
        edge_xs.append(start_pos[0])
        edge_xs.append(end_pos[0])
        edge_xs.append(None)
        edge_ys.append(start_pos[1])
        edge_ys.append(end_pos[1])
        edge_ys.append(None)
        edge_zs.append(start_pos[2])
        edge_zs.append(end_pos[2])
        edge_zs.append(None)

  f.add_trace(go.Scatter3d(
    x=edge_xs,
    y=edge_ys,
    z=edge_zs,
    name='SOM edges',
    mode='lines',
    marker=dict(
      color='#636efa',
    )
  ))


  node_xs = []
  node_ys = []
  node_zs = []
  for row in grid:
    for vec in row:
      node_xs.append(vec[0])
      node_ys.append(vec[1])
      node_zs.append(vec[2])
  
  f.add_trace(go.Scatter3d(
    x=node_xs,
    y=node_ys,
    z=node_zs,
    name='SOM nodes',
    mode='markers',
    marker=dict(
      color='#ef553b',
      size=3,
    )
  ))

  f.update_layout(
    title=dict(
      text="Fit a self-organizing map to the data"
    )
  )

  f.show()
  if save:
    f.write_html("plot_fit.html", include_plotlyjs='cdn')


def plot_rendered_som(data, som, save=False):
  som_grid_size = som.get_grid_size()

  preds = np.zeros((som_grid_size, som_grid_size)).tolist()
  for d in data:
    i, j = som.classify(d)
    preds[i][j] += 1

  f = px.imshow(
    preds,
    labels=dict(
      color='Count'
    ),
    color_continuous_scale='Mint'
  )

  f.update_layout(
    xaxis=dict(
      tickvals=[],
      ticktext=[],
    ),
    yaxis=dict(
      tickvals=[],
      ticktext=[],
    ),
    title=dict(
      text="Cluster the data according to the closest node in the self-organizing map"
    )
  )

  f.show()
  if save:
    f.write_html("plot_grid.html", include_plotlyjs='cdn')

  


def main():
  dim = 3 
  #data = gen_clustered_vectors(1000, dim)
  #data = gen_4_corners(1000)
  data = gen_3d_clusters(100)
  # maybe use penguins for the data? idk
  som_grid_size = 10
  som = SOM(grid_size=som_grid_size, vec_size=dim)
  som.train(data, epochs=20)


  #plot_2d(data, som)
  plot_3d(data, som, save=False)
  plot_rendered_som(data, som, save=False)
  return

      
if __name__ == '__main__':
  main()
