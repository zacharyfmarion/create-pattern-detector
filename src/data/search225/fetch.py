"""Polite sequential client for the ExplOri 22.5 ``/api/fetch_tiling`` endpoint.

The site is a local-style ``ThreadingHTTPServer`` backed by SQLite + per-request
reconstruction (see SEARCH-22.5's ``database/tilings/inspect.py::pull_specific_tiling``).
It is *not* safe to hammer concurrently — parallel requests collide in the per-request
SQLAlchemy session and the server returns ``400``. We therefore fetch strictly
sequentially with a small delay, which is also the courteous thing to do.

The endpoint takes three params:
    id   -- the tiling id within a (N, sym) database (sequential, with gaps)
    N    -- grid size (2..6 are populated)
    sym  -- one of {"none", "diag", "book"}

The ``view?id=`` URLs on the site encode the same thing as ``{N}{symchar}{id}`` where
symchar is n/b/d (none/book/diag), e.g. ``4d129102`` == N=4, diag, id=129102.

Vertex coordinates are stored as 8 integers ``[a,b,c,d,e,f,g,h]`` representing the
exact value ``a/b + (sqrt2/2)*(c/d - g/h)`` (x) and ``e/f + (sqrt2/2)*(c/d + g/h)`` (y).
This mirrors SEARCH-22.5's own ``utils.exportFold`` / ``_vertex_to_xy``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Iterator

import requests

BASE_URL = "https://225.designorigami.net"

# The site rejects requests without a browser-like User-Agent (Cloudflare 403).
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

SQRT2_OVER_2 = math.sqrt(2.0) / 2.0

# Symmetry name -> single-char code used in the site's ``view?id=`` URLs.
SYM_TO_CHAR = {"none": "n", "book": "b", "diag": "d"}


@dataclass(frozen=True)
class Combo:
    """A populated (grid size, symmetry) database on the site."""

    N: int
    sym: str

    @property
    def key(self) -> str:
        return f"{self.N}_{self.sym}"

    @property
    def char(self) -> str:
        return SYM_TO_CHAR[self.sym]

    def view_id(self, tiling_id: int) -> str:
        """The id string used in ``/view?id=...`` URLs (e.g. ``4d129102``)."""
        return f"{self.N}{self.char}{tiling_id}"


# Combos that actually have data on the site. N=2 has only a few dozen trivial
# patterns; N>=3 is where the useful volume lives. (N=5/diag is the largest, ~1M.)
ALL_COMBOS: list[Combo] = [
    Combo(2, "none"), Combo(2, "diag"), Combo(2, "book"),
    Combo(3, "none"), Combo(3, "diag"), Combo(3, "book"),
    Combo(4, "none"), Combo(4, "diag"), Combo(4, "book"),
    Combo(5, "diag"), Combo(5, "book"),
    Combo(6, "book"),
]

# Approximate upper bound on the id range for each combo, established by probing the
# live API (the boundary between 200 and 404). IDs are sequential with small gaps, so
# these are used directly as the random-sampling ceiling: over-estimating just yields a
# few more "missing id" draws (which are retried), and the builder's miss-limit guards
# against a wildly wrong value. This avoids paying for runtime discovery, where every
# existence probe downloads the full ~275KB tiling payload.
MAX_ID_HINTS: dict[str, int] = {
    "2_none": 40, "2_diag": 30, "2_book": 20,
    "3_none": 150_000, "3_diag": 2_000, "3_book": 600,
    "4_none": 400_000, "4_diag": 210_000, "4_book": 90_000,
    "5_diag": 1_100_000, "5_book": 200_000,
    "6_book": 4_700,
}


def vertex_to_cartesian(v: list[float]) -> list[float]:
    """Convert an 8-element exact vertex to ``[x, y]`` float coordinates.

    Mirrors SEARCH-22.5's ``exportFold``: ``x = a/b + s*(c/d - g/h)`` and
    ``y = e/f + s*(c/d + g/h)`` with ``s = sqrt(2)/2``.
    """
    vx = v[0] / v[1] if v[1] else 0.0
    vy = v[2] / v[3] if v[3] else 0.0
    vz = v[4] / v[5] if v[5] else 0.0
    vw = v[6] / v[7] if v[7] else 0.0
    return [
        vx + SQRT2_OVER_2 * (vy - vw),
        vz + SQRT2_OVER_2 * (vy + vw),
    ]


def edge_assignment_label(raw_type: Any) -> str:
    """Map a SEARCH-22.5 edge type to a FOLD assignment (M/V/B/F).

    Mirrors ``getFoldType`` in the site's ``utils.js``. Border -> B, mountain
    variants -> M, valley variants -> V, auxiliary/hinge/flat -> F.
    """
    if not raw_type:
        return "F"
    t = str(raw_type).strip().lower()
    if t == "b":
        return "B"
    if t in ("rm", "m", "hm"):
        return "M"
    if t in ("rv", "av", "v", "hv"):
        return "V"
    if t in ("h", "aux", "ax"):
        return "F"
    if "m" in t:
        return "M"
    if "v" in t:
        return "V"
    return "F"


def cp_to_fold_dict(cp: dict, combo: Combo, tiling_id: int) -> dict:
    """Convert the API ``cp`` object into a minimal, valid FOLD dictionary."""
    vertices_coords = [vertex_to_cartesian(v) for v in cp["vertices"]]
    edges_vertices: list[list[int]] = []
    edges_assignment: list[str] = []
    for edge in cp["edges"]:
        edges_vertices.append([int(edge[0]), int(edge[1])])
        edges_assignment.append(edge_assignment_label(edge[2] if len(edge) > 2 else None))
    return {
        "file_spec": 1.1,
        "file_creator": "SEARCH-22.5",
        "file_classes": ["creasePattern"],
        "file_source": f"{BASE_URL}/view?id={combo.view_id(tiling_id)}",
        "vertices_coords": vertices_coords,
        "edges_vertices": edges_vertices,
        "edges_assignment": edges_assignment,
    }


class TilingNotFound(Exception):
    """Raised when an id is absent (400/404) — the range may continue."""


class TilingClient:
    """Sequential, retrying client for the fetch_tiling endpoint."""

    def __init__(
        self,
        *,
        base_url: str = BASE_URL,
        delay: float = 0.2,
        timeout: float = 30.0,
        retries: int = 3,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.delay = delay
        self.timeout = timeout
        self.retries = retries
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

    def _get(self, tiling_id: int, combo: Combo) -> requests.Response:
        url = f"{self.base_url}/api/fetch_tiling"
        params = {"id": tiling_id, "N": combo.N, "sym": combo.sym}
        last_exc: Exception | None = None
        for attempt in range(self.retries + 1):
            if self.delay:
                time.sleep(self.delay if attempt == 0 else self.delay * (attempt + 1))
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
            except requests.RequestException as exc:  # transient network error
                last_exc = exc
                continue
            # 400/404 mean "this id doesn't exist", not a transient failure.
            if resp.status_code in (400, 404):
                raise TilingNotFound(combo.view_id(tiling_id))
            if resp.status_code >= 500:  # server hiccup -> retry
                last_exc = requests.HTTPError(f"{resp.status_code} for {combo.view_id(tiling_id)}")
                continue
            resp.raise_for_status()
            return resp
        raise requests.HTTPError(
            f"Failed after {self.retries + 1} attempts for {combo.view_id(tiling_id)}: {last_exc}"
        )

    def fetch_cp(self, tiling_id: int, combo: Combo) -> dict | None:
        """Return the ``cp`` dict for a tiling, or None if it exists but is empty.

        Raises ``TilingNotFound`` if the id is absent.
        """
        resp = self._get(tiling_id, combo)
        data = resp.json()
        results = data.get("results") or []
        if not results:
            raise TilingNotFound(combo.view_id(tiling_id))
        cp = results[0].get("cp")
        if not cp or not cp.get("vertices") or not cp.get("edges"):
            return None
        return cp

    def fetch_fold(self, tiling_id: int, combo: Combo) -> dict | None:
        """Fetch a tiling and return a FOLD dict, or None if empty/malformed."""
        cp = self.fetch_cp(tiling_id, combo)
        if cp is None:
            return None
        return cp_to_fold_dict(cp, combo, tiling_id)

    def exists(self, tiling_id: int, combo: Combo) -> bool:
        """Cheap existence check used by max-id discovery."""
        try:
            self._get(tiling_id, combo)
            return True
        except TilingNotFound:
            return False

    def discover_max_id(self, combo: Combo, *, ceiling: int = 4_000_000) -> int:
        """Find an id near the top of a combo's range via exponential + binary search.

        Prefer ``MAX_ID_HINTS`` (cheap, no requests). Only fall back to live probing
        for combos without a hint. Note each probe downloads the full tiling payload,
        so this is deliberately a fallback, not the default path.

        IDs are sequential with small gaps, so the result is an *upper-ish* bound
        suitable for random sampling, not a guaranteed exact maximum.
        """
        hint = MAX_ID_HINTS.get(combo.key)
        if hint is not None:
            return hint

        def any_exists_near(center: int, window: int = 12) -> bool:
            for offset in range(window):
                cand = center - offset
                if cand < 1:
                    break
                if self.exists(cand, combo):
                    return True
            return False

        # Exponential growth until we overshoot the populated range.
        low, high = 1, 1
        while high < ceiling and any_exists_near(high):
            low = high
            high = min(high * 4, ceiling)
        # Binary search for the boundary between "populated" and "empty".
        while low + 1 < high:
            mid = (low + high) // 2
            if any_exists_near(mid):
                low = mid
            else:
                high = mid
        return low


def iter_sequential_ids(start: int = 1) -> Iterator[int]:
    """Yield ids 1, 2, 3, ... for full sequential mirroring (not used by sampling)."""
    n = start
    while True:
        yield n
        n += 1
