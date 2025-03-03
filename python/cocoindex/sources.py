"""All builtin sources."""
from . import op

class LocalFile(op.SourceSpec):
    """Import data from local file system."""

    _op_category = op.OpCategory.SOURCE

    path: str
    binary: bool = False
