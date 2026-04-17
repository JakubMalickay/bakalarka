"""Export cube metadata (dimensions and levels) to JSON using OLAPConnection.

Usage:
- Set env vars OLAP_SERVER, OLAP_CATALOG, etc., or rely on the AdventureWorks defaults.
- Run: python olap_metadata_export.py
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

from OLAP_connection import OLAPConnection, build_default_awdw, build_from_env


def cube_metadata(conn: OLAPConnection, cube_name: str) -> Dict[str, Any]:
    dims = conn.list_dimensions(cube_name)
    dimensions = []
    for name, unique_name, description in dims:
        raw_levels = conn.list_levels(cube_name, unique_name)
        levels = []
        attributes = []
        for r in raw_levels:
            level_name = r[0]
            level_unique = r[1]
            level_desc = r[2]
            level_origin = r[3] if len(r) > 3 else None
            name_upper = (level_name or "").upper()
            if any(tag in name_upper for tag in ["ALL", "KEY", "SORT", "ID"]):
                continue
            full_desc = f"Parent dimension: {name}. {level_desc or level_unique}"
            if level_origin == 2:  # attribute hierarchy
                attributes.append(
                    {
                        "type": "attribute",
                        "name": level_name,
                        "unique_name": level_unique,
                        "description": full_desc,
                    }
                )
            else:
                levels.append(
                    {
                        "type": "level",
                        "name": level_name,
                        "unique_name": level_unique,
                        "description": full_desc,
                    }
                )
        dimensions.append(
            {
                "type": "dimension",
                "name": name,
                "unique_name": unique_name,
                "description": description,
                "levels": levels,
                "attributes": attributes,
            }
        )
    return {"type": "cube", "cube": cube_name, "dimensions": dimensions}


def measures_metadata(conn: OLAPConnection, cube_name: str) -> Dict[str, Any]:
    measures = [
        {
            "type": "measure",
            "name": r[0],
            "unique_name": r[1],
            "measure_group": r[2],
            "description": r[0] or r[1],
        }
        for r in conn.list_measures(cube_name)
    ]
    return {"type": "cube", "cube": cube_name, "measures": measures}


def main() -> None:
    try:
        conn = build_from_env()
    except Exception:
        conn = build_default_awdw()

    cubes = conn.list_cubes()
    cube_meta = [cube_metadata(conn, cube) for cube in cubes]
    measures_meta = [measures_metadata(conn, cube) for cube in cubes]

    print("Dimensions + levels:")
    print(json.dumps(cube_meta, indent=2))

    with open("cube_dimensions.json", "w", encoding="utf-8") as f:
        json.dump(cube_meta, f, indent=2)

    print("\nMeasures:")
    print(json.dumps(measures_meta, indent=2))

    with open("cube_measures.json", "w", encoding="utf-8") as f:
        json.dump(measures_meta, f, indent=2)


if __name__ == "__main__":
    main()
