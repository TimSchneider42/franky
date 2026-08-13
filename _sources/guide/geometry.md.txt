# 🧮 Geometry

`franky.Affine` is a python wrapper for [Eigen::Affine3d](https://eigen.tuxfamily.org/dox/group__TutorialGeometry.html).
It is used for Cartesian poses, frames and transformation.
franky adds its own constructor, which takes a position and a quaternion as inputs:

```python
import math
from scipy.spatial.transform import Rotation
from franky import Affine

z_translation = Affine([0.0, 0.0, 0.5])

quat = Rotation.from_euler("xyz", [0, 0, math.pi / 2]).as_quat()
z_rotation = Affine([0.0, 0.0, 0.0], quat)

combined_transformation = z_translation * z_rotation
```

In all cases, distances are in [m] and rotations in [rad].
