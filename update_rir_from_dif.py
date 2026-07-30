#!/usr/bin/env python3
"""Update RIR and density on existing minerals from AMCSD bulk DIF."""

import argparse
from pathlib import Path
from utils.local_database import LocalCIFDatabase


def main():
    parser = argparse.ArgumentParser(description="Update mineral RIR values from AMCSD DIF")
    parser.add_argument(
        "--dif",
        default=str(Path(__file__).parent / "data" / "difdata.dif"),
        help="Path to AMCSD bulk DIF file",
    )
    parser.add_argument(
        "--db",
        default=None,
        help="Optional path to local_cif_database.db",
    )
    args = parser.parse_args()

    db = LocalCIFDatabase(args.db)
    stats = db.update_rir_from_dif(args.dif)
    print(stats)


if __name__ == "__main__":
    main()
