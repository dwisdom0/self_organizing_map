This is a quick implementation of a [self-organizing map](https://en.wikipedia.org/wiki/Self-organizing_map).
You can read [my blog post about it](dwisdom.gitlab.io/post/self_organizing_map).


`som.py` is my first pass at the implementation.
It's a classic example of needing to build things at least two times (probably three, to be honest) if you really want to get it right.
`som_numpy.py` is my second attempt.
There are still a few things to improve, but it's fast enough to generate some pretty plots for my blog.
That's all I need it for.

# Quickstart
```shell
pip install -r requirements.txt
python plot_som_grid.py
```
