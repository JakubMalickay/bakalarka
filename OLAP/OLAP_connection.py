"""Lightweight OLAP connection helper for reuse in other projects.

Depends on pyadomd and ADOMD.NET client. If ADOMD.NET is installed to a
non-default path, set ADOMD_PATH or adjust the path append below.
"""
from __future__ import annotations

import os
from typing import Any, Iterable, List, Optional, Sequence, Tuple

# Make sure ADOMD.NET provider is on sys.path
ADOMD_PATH = os.environ.get("ADOMD_PATH", r"C:\\Program Files\\Microsoft.NET\\ADOMD.NET\\160")
if ADOMD_PATH and ADOMD_PATH not in os.sys.path:
    os.sys.path.append(ADOMD_PATH)

from pyadomd import Pyadomd  # type: ignore


class OLAPConnection:
    """Tiny wrapper around pyadomd for SSAS/OLAP connectivity."""

    def __init__(
        self,
        server: str,
        catalog: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
        integrated_security: bool = True,
        provider: str = "MSOLAP",
    ) -> None:
        self.server = server
        self.catalog = catalog
        self.username = username
        self.password = password
        self.integrated_security = integrated_security
        self.provider = provider
        self.conn_str = self._build_conn_str()

    def _build_conn_str(self) -> str:
        parts = [
            f"Provider={self.provider}",
            f"Data Source={self.server}",
            f"Catalog={self.catalog}",
        ]
        if self.integrated_security:
            parts.append("Integrated Security=SSPI")
        else:
            if not self.username or not self.password:
                raise ValueError("Username and password required when Integrated Security is False")
            parts.append(f"User ID={self.username}")
            parts.append(f"Password={self.password}")
            parts.append("Persist Security Info=True")
        return ";".join(parts) + ";"

    def test_connection(self) -> bool:
        """Return True if a basic metadata query succeeds."""
        try:
            with Pyadomd(self.conn_str) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT * FROM $SYSTEM.DBSCHEMA_CATALOGS")
                    cur.fetchall()
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"OLAP test connection failed: {exc}")
            return False

    def list_cubes(self) -> List[str]:
        """List visible cubes (excluding perspectives with CUBE_SOURCE != 1)."""
        query = (
            "SELECT [CUBE_NAME] FROM $SYSTEM.MDSCHEMA_CUBES WHERE [CUBE_SOURCE] = 1"
        )
        rows, _ = self.execute_mdx(query)
        return [r[0] for r in rows]

    def execute_mdx(self, mdx: str) -> Tuple[List[Tuple[Any, ...]], Sequence[str]]:
        """Execute MDX and return (rows, headers)."""
        with Pyadomd(self.conn_str) as conn:
            with conn.cursor() as cur:
                cur.execute(mdx)
                headers = [col[0] for col in cur.description]
                rows = cur.fetchall()
                return rows, headers

    def list_dimensions(self, cube_name: str) -> List[Tuple[Any, ...]]:
        mdx = (
            "SELECT [DIMENSION_NAME],[DIMENSION_UNIQUE_NAME],[DESCRIPTION] "
            "FROM $SYSTEM.MDSCHEMA_DIMENSIONS "
            f"WHERE [CUBE_NAME] = '{cube_name}' AND [DIMENSION_UNIQUE_NAME] <> '[Measures]'"
        )
        rows, _ = self.execute_mdx(mdx)
        return rows

    def list_measures(self, cube_name: str) -> List[Tuple[Any, ...]]:
        mdx = (
            "SELECT [MEASURE_NAME],[MEASURE_UNIQUE_NAME],[MEASUREGROUP_NAME] "
            "FROM $SYSTEM.MDSCHEMA_MEASURES "
            f"WHERE [CUBE_NAME] = '{cube_name}'"
        )
        rows, _ = self.execute_mdx(mdx)
        return rows

    def list_levels(self, cube_name: str, dimension_unique_name: str) -> List[Tuple[Any, ...]]:
        mdx = (
            "SELECT [LEVEL_NAME],[LEVEL_UNIQUE_NAME],[DESCRIPTION],[LEVEL_ORIGIN] "
            "FROM $SYSTEM.MDSCHEMA_LEVELS "
            f"WHERE [CUBE_NAME] = '{cube_name}' "
            f"AND [DIMENSION_UNIQUE_NAME] = '{dimension_unique_name}'"
        )
        rows, _ = self.execute_mdx(mdx)
        return rows


# Defaults for the AdventureWorks demo (AWDW2019Multidimensional-SE) on 68.210.184.147 using SSPI
DEFAULT_SERVER = "68.210.184.147"
DEFAULT_CATALOG = "AWDW2019Multidimensional-SE"
DEFAULT_INTEGRATED = True


def build_default_awdw() -> OLAPConnection:
    """Build connection using the local AdventureWorks demo values."""
    return OLAPConnection(
        DEFAULT_SERVER,
        DEFAULT_CATALOG,
        integrated_security=DEFAULT_INTEGRATED,
    )


def build_from_env(prefix: str = "OLAP") -> OLAPConnection:
    """Convenience constructor using env vars like OLAP_SERVER, OLAP_CATALOG."""
    server = os.environ.get(f"{prefix}_SERVER")
    catalog = os.environ.get(f"{prefix}_CATALOG")
    username = os.environ.get(f"{prefix}_USER")
    password = os.environ.get(f"{prefix}_PASSWORD")
    integrated = os.environ.get(f"{prefix}_INTEGRATED", "true").lower() == "true"
    if not server or not catalog:
        raise ValueError(f"Set {prefix}_SERVER and {prefix}_CATALOG environment variables")
    return OLAPConnection(server, catalog, username, password, integrated)


if __name__ == "__main__":
    # Example: use the default AdventureWorks connection
    conn = build_default_awdw()
    print("Connection OK?", conn.test_connection())
    print("Cubes:", conn.list_cubes())
