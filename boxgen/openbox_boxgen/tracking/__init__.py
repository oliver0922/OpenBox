"""AB3DMOT tracking (stage 3): ``track_per_class`` runs ``AB3DMOT`` forward and backward per category.

This package is derived from AB3DMOT by Xinshuo Weng (Carnegie Mellon University,
https://github.com/xinshuoweng/AB3DMOT): ``box``, ``dist_metrics``, ``kalman_filter``,
``matching`` and ``model`` follow the original modules, adapted to the Waymo frame,
per-class settings and a forward + backward pass.  AB3DMOT is released under a
non-commercial research licence, reproduced in ``LICENSE_AB3DMOT`` next to this
file; this derivative is subject to the same terms.

Requires ``filterpy`` (1.4.5 produced the release), ``scipy`` and ``numpy``;
``track_per_class`` also needs torch and the built pcdet ops through
``openbox_boxgen.boxes``.
"""
from .model import AB3DMOT
from .track_per_class import track_per_class

__all__ = ['AB3DMOT', 'track_per_class']
