"""Allow `python -m occlude` invocation."""
import sys

from occlude.cli import main

if __name__ == "__main__":
    sys.exit(main())
