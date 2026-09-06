"""Neighbour queries, using scipy's KD-tree when it is installed.

The samplers need three things from a point set: the k nearest within a
per-point radius, the plain nearest, and a per-point radius search. cKDTree
does all three across every core; without it a uniform grid does the same work
in numpy. The grid is exact. The two backends disagree only on the last ulp of
a distance, which reaches the finished marks at 1e-16 relative.
"""

from __future__ import annotations

import numpy as np

__all__ = ["neighbours", "HAVE_SCIPY"]

try:
    from scipy.spatial import cKDTree
    HAVE_SCIPY = True
except ImportError:                     # pragma: no cover
    cKDTree = None
    HAVE_SCIPY = False

#: Cap on grid cells, so a degenerate point set cannot allocate an enormous
#: index. Exceeding it coarsens the cell instead.
MAX_CELLS = 4_000_000


class _KDTreeIndex:
    def __init__(self, pts: np.ndarray):
        self.data = np.ascontiguousarray(pts, dtype=np.float64)
        self._t = cKDTree(self.data)

    def knn_within(self, radius: np.ndarray, k: int):
        d, i = self._t.query(self.data, k=k + 1, workers=-1)
        d, i = d[:, 1:], i[:, 1:]
        return np.where(d <= radius[:, None], d, np.inf), i

    def nearest(self, q: np.ndarray):
        return self._t.query(np.asarray(q, dtype=np.float64), workers=-1)

    def ball(self, q: np.ndarray, radii: np.ndarray):
        return self._t.query_ball_point(np.asarray(q, dtype=np.float64),
                                        radii, workers=-1)


class _GridIndex:
    """Uniform bucket grid over the point set.

    Cells hold about one point each, so a search radius of one cell needs
    only the ring around the query's own cell. Wider radii take a wider ring;
    queries are grouped by the ring they need, one gather per group.
    """

    def __init__(self, pts: np.ndarray):
        self.data = np.ascontiguousarray(pts, dtype=np.float64)
        n = len(self.data)

        if n:
            self.lo = self.data.min(axis=0)
            span = np.maximum(self.data.max(axis=0) - self.lo, 1e-9)
        else:
            self.lo = np.zeros(2)
            span = np.ones(2)

        cell = float(np.sqrt(span[0] * span[1] / max(n, 1)))
        cell = max(cell, float(span.max()) / 65535.0, 1e-9)
        if (span[0] / cell + 1) * (span[1] / cell + 1) > MAX_CELLS:
            cell = float(np.sqrt(span[0] * span[1] / MAX_CELLS))
        self.cell = cell

        self.gw = int(span[0] / cell) + 1
        self.gh = int(span[1] / cell) + 1
        self.reach = self.gw + self.gh          # ring that covers the whole grid

        cx, cy = self.cells(self.data)
        cid = cy * self.gw + cx
        self.order = np.argsort(cid, kind="stable")
        self.counts = np.bincount(cid, minlength=self.gw * self.gh)
        self.starts = np.concatenate([[0], np.cumsum(self.counts)[:-1]])

    def cells(self, pts: np.ndarray):
        c = (np.asarray(pts, dtype=np.float64) - self.lo) / self.cell
        cx = np.clip(c[:, 0].astype(np.int64), 0, self.gw - 1)
        cy = np.clip(c[:, 1].astype(np.int64), 0, self.gh - 1)
        return cx, cy

    def _gather(self, qx: np.ndarray, qy: np.ndarray, r: int):
        """Candidate pairs for a block of (2r+1)^2 cells around each query.

        Returns (qi, pi): qi indexes into qx/qy, pi into self.data. The
        per-cell runs of `order` are expanded unpadded, so cost tracks points
        found, not cells looked at.
        """
        off = np.arange(-r, r + 1, dtype=np.int64)
        ox = np.repeat(off, off.size)
        oy = np.tile(off, off.size)
        b = ox.size

        qi_out, pi_out = [], []
        # Cap the (queries x cells) intermediate rather than the output.
        chunk = max(1, int(4_000_000 // b))
        for s in range(0, qx.size, chunk):
            ax = qx[s:s + chunk, None] + ox[None, :]
            ay = qy[s:s + chunk, None] + oy[None, :]
            ok = (ax >= 0) & (ax < self.gw) & (ay >= 0) & (ay < self.gh)
            cid = np.where(ok, ay * self.gw + ax, 0)
            cnt = np.where(ok, self.counts[cid], 0).ravel()
            st = self.starts[cid].ravel()

            total = int(cnt.sum())
            if total == 0:
                continue
            blk = np.repeat(np.arange(cnt.size), cnt)
            pos = np.arange(total) - (np.cumsum(cnt) - cnt)[blk]
            qi_out.append(s + blk // b)
            pi_out.append(self.order[st[blk] + pos])

        if not qi_out:
            return np.empty(0, np.int64), np.empty(0, np.int64)
        return np.concatenate(qi_out), np.concatenate(pi_out)

    def _dist(self, q: np.ndarray, qi: np.ndarray, pi: np.ndarray):
        # sqrt of the sum of squares rather than hypot: 1.6x faster measured,
        # and canvas coordinates are nowhere near hypot's overflow range.
        dx = q[qi, 0] - self.data[pi, 0]
        dy = q[qi, 1] - self.data[pi, 1]
        return np.sqrt(dx * dx + dy * dy)

    def _rings(self, radius: np.ndarray):
        """Ring width each query needs, as (width, query indices) groups.

        The spacing field is infinite where density is zero and rounding can
        land there, so an infinite radius reaches this point. It means every
        point, i.e. the ring that spans the grid.
        """
        r = np.asarray(radius, dtype=np.float64) / self.cell
        r = np.where(np.isfinite(r), np.ceil(r), float(self.reach))
        r = np.clip(r, 1, self.reach).astype(np.int64)
        for w in np.unique(r):
            yield int(w), np.nonzero(r == w)[0]

    def knn_within(self, radius: np.ndarray, k: int):
        n = len(self.data)
        dist = np.full((n, k), np.inf)
        idx = np.zeros((n, k), dtype=np.int64)
        if n == 0 or k <= 0:
            return dist, idx

        qx, qy = self.cells(self.data)
        for width, sel in self._rings(radius):
            qi, pi = self._gather(qx[sel], qy[sel], width)
            if qi.size == 0:
                continue
            q = sel[qi]
            d = self._dist(self.data, q, pi)
            keep = (pi != q) & (d <= radius[q])
            q, pi, d = q[keep], pi[keep], d[keep]
            if q.size == 0:
                continue

            # Sorted by query then distance, so position within a run is rank.
            o = np.lexsort((d, q))
            q, pi, d = q[o], pi[o], d[o]
            head = np.ones(q.size, dtype=bool)
            head[1:] = q[1:] != q[:-1]
            start = np.nonzero(head)[0]
            rank = np.arange(q.size) - np.repeat(start, np.diff(np.append(start, q.size)))

            m = rank < k
            dist[q[m], rank[m]] = d[m]
            idx[q[m], rank[m]] = pi[m]
        return dist, idx

    def nearest(self, q: np.ndarray):
        q = np.asarray(q, dtype=np.float64)
        m = len(q)
        best_d = np.full(m, np.inf)
        best_i = np.zeros(m, dtype=np.int64)
        if m == 0 or len(self.data) == 0:
            return best_d, best_i

        qx, qy = self.cells(q)
        todo = np.arange(m)
        width = 1
        while todo.size:
            qi, pi = self._gather(qx[todo], qy[todo], width)
            if qi.size:
                g = todo[qi]
                d = self._dist(q, g, pi)
                o = np.lexsort((d, g))
                g, d, pi = g[o], d[o], pi[o]
                head = np.ones(g.size, dtype=bool)
                head[1:] = g[1:] != g[:-1]
                g, d, pi = g[head], d[head], pi[head]
                better = d < best_d[g]
                best_d[g[better]] = d[better]
                best_i[g[better]] = pi[better]

            if width >= self.reach:
                break
            # A hit inside width*cell cannot be beaten from outside the block.
            todo = todo[~(best_d[todo] <= width * self.cell)]
            width = min(width * 2, self.reach)
        return best_d, best_i

    def ball(self, q: np.ndarray, radii):
        q = np.asarray(q, dtype=np.float64)
        radii = np.broadcast_to(np.asarray(radii, dtype=np.float64), (len(q),))
        out: list[list[int]] = [[] for _ in range(len(q))]
        if len(q) == 0 or len(self.data) == 0:
            return out

        qx, qy = self.cells(q)
        for width, sel in self._rings(radii):
            qi, pi = self._gather(qx[sel], qy[sel], width)
            if qi.size == 0:
                continue
            g = sel[qi]
            d = self._dist(q, g, pi)
            keep = d <= radii[g]
            g, pi = g[keep], pi[keep]
            if g.size == 0:
                continue
            o = np.argsort(g, kind="stable")
            g, pi = g[o], pi[o]
            counts = np.bincount(g, minlength=len(q))
            parts = np.split(pi, np.cumsum(counts)[:-1])
            for j in sel:
                out[j] = parts[j].tolist()
        return out


def neighbours(pts: np.ndarray):
    """Index a point set for neighbour queries."""
    return _KDTreeIndex(pts) if HAVE_SCIPY else _GridIndex(pts)
