"""lc0 HTTP client with concise output formatting for the LLM context."""
from __future__ import annotations
import httpx


class LcOClient:
    def __init__(self, base_url: str = "http://localhost:7100"):
        self._url = base_url.rstrip("/")
        self._http = httpx.Client(timeout=60.0)

    def health(self) -> dict:
        return self._http.get(f"{self._url}/health").raise_for_status().json()

    def analyze(self, fen: str, nodes: int = 800, multipv: int = 3) -> dict:
        r = self._http.post(
            f"{self._url}/analyze",
            json={"fen": fen, "nodes": nodes, "multipv": multipv},
        )
        r.raise_for_status()
        data = r.json()
        lines = []
        for pv in data.get("multipv", []):
            if pv.get("mate") is not None:
                score = f"M{pv['mate']}"
            else:
                score = f"{(pv.get('score_cp') or 0) / 100:+.2f}"
            moves = " ".join(pv.get("pv", [])[:6])
            lines.append(
                f"{pv['multipv']}. {moves}  score={score}  depth={pv.get('depth')}"
            )
        return {
            "turn": data.get("turn"),
            "bestmove": data.get("bestmove"),
            "lines": lines,
        }

    def get_policy(self, fen: str, nodes: int | None = None) -> dict:
        payload: dict = {"fen": fen}
        if nodes is not None:
            payload["nodes"] = nodes
        r = self._http.post(f"{self._url}/policy", json=payload)
        r.raise_for_status()
        data = r.json()
        moves = sorted(
            data.get("moves", []),
            key=lambda m: m.get("P") or 0,
            reverse=True,
        )[:12]
        rows = []
        for m in moves:
            p = f"{(m['P'] or 0)*100:.1f}%" if m.get("P") is not None else "?"
            v = f"{m['V']:+.3f}" if m.get("V") is not None else "?"
            q = f"{m['Q']:+.3f}" if m.get("Q") is not None else "?"
            rows.append(f"{m['move']}  P={p}  V={v}  Q={q}  N={m.get('N', 0)}")
        return {
            "turn": data.get("turn"),
            "value_root": round(data.get("value_root") or 0, 4),
            "nodes_searched": data.get("nodes_searched"),
            "top_moves": rows,
        }

    def close(self):
        self._http.close()
