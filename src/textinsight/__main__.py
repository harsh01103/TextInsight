"""Allow ``python -m textinsight``, which behaves like the ``textinsight`` command."""

from .cli import main

raise SystemExit(main())
