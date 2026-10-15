"""Allow `python -m applefoundationmodels` as an alias for the `afm` CLI."""

import sys

from .cli import main

sys.exit(main())
