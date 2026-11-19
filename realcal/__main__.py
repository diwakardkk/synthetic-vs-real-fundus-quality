"""Allow ``python -m realcal <stage>`` as a shorthand for ``python -m realcal.cli <stage>``."""
from realcal.cli import main

raise SystemExit(main())
