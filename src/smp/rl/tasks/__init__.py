"""SMP downstream RL tasks.

Importing this package registers all SMP tasks in ``mjlab.tasks.registry``
via side-effect imports of each task sub-package.
"""

from smp.rl.tasks import (
  getup,  # noqa: F401  # registers G1/X2 getup tasks
  location,  # noqa: F401  # registers G1/X2 location tasks
  steering,  # noqa: F401  # registers G1/X2 steering and forward tasks
)
