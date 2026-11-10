"""HTTP adapter over the atlas: brain data in binary now, streamed chat in a later commit.

The atlas loads ONCE, in the lifespan, and lives on ``app.state``.
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Response

from flybrain.atlas import RowIndex, load_atlas
from flybrain.mcp_server import DEFAULT_ANNOTATIONS

BYTES_PER_POINT = 12  # 3 x float32


def create_app(atlas_path=None) -> FastAPI:
    path = Path(atlas_path or os.environ.get("FLYBRAIN_ANNOTATIONS") or DEFAULT_ANNOTATIONS)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        atlas = load_atlas(path)
        app.state.atlas = atlas
        app.state.row_index = RowIndex(atlas)
        # little-endian float32, xyz interleaved, atlas row order: GPU-ready, built once
        xyz = atlas.neurons[["x_um", "y_um", "z_um"]].to_numpy()
        app.state.points = np.ascontiguousarray(xyz.astype("<f4")).tobytes()
        yield

    app = FastAPI(title="flybrain", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": "ok", "neurons": len(app.state.atlas.neurons)}

    @app.get("/brain/meta")
    def brain_meta():
        count = len(app.state.atlas.neurons)
        return {
            "count": count,
            "dtype": "float32",
            "endianness": "little",
            "layout": "xyz-interleaved",
            "order": "atlas rows",
            "extent_um": list(app.state.atlas.extent_um()),
            "bytes": count * BYTES_PER_POINT,
        }

    @app.get("/brain/points")
    def brain_points():
        return Response(
            app.state.points,
            media_type="application/octet-stream",
            headers={"Cache-Control": "public, max-age=3600"},
        )

    return app
