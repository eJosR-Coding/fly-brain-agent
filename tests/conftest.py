import pytest

from flybrain.atlas import Atlas, load_atlas

@pytest.fixture(autouse=True)
def _no_real_connectome(monkeypatch, tmp_path):
    """Tests never pick up a developer's real data/processed/connectome_783.npz."""
    monkeypatch.setenv("FLYBRAIN_CONNECTOME", str(tmp_path / "no-connectome.npz"))


HEADER = "root_id\tpos_x\tpos_y\tpos_z\tsuper_class\tcell_class\tcell_type\ttop_nt\tside\tflow"


def _tiny_rows() -> list[str]:
    rows = [f"{i}\t1000\t2000\t100\tcentral\tCX\tPEN_a\tacetylcholine\tleft\tintrinsic" for i in range(1, 61)]
    rows += [
        "100\t3000\t2500\t200\tcentral\tCX\tEPG\tglutamate\tright\tintrinsic",
        "101\t500\t100\t50\toptic\tOL\tT4a\tacetylcholine\tleft\tafferent",
        "102\t200\t300\t10\tsensory\t\tORN_DA1\t\tleft\tafferent",
    ]
    return rows


@pytest.fixture
def tiny_tsv_path(tmp_path):
    """Path to the tiny annotations TSV (63 neurons)."""
    path = tmp_path / "ann.tsv"
    path.write_text("\n".join([HEADER, *_tiny_rows()]) + "\n")
    return path


@pytest.fixture
def tiny_atlas(tiny_tsv_path) -> Atlas:
    """Tiny atlas: 60 central acetylcholine neurons + a few others."""
    return load_atlas(tiny_tsv_path)


# Tiny structural graph over tiny_atlas rows (root_id i -> row i-1 for 1..60; 100 -> 60 EPG,
# 101 -> 61 T4a, 102 -> 62 ORN_DA1):
#   EPG(60) -> 0 -> 1 -> 2,  1 -> 3 -> 2,  T4a(61) -> EPG, ORN(62) -> T4a,
#   cycle 5 -> 6 -> 7 -> 5, weak edge 0 -> 8 (2 synapses, under the threshold)
TINY_EDGES = [
    (60, 0, 10), (0, 1, 10), (1, 2, 10), (1, 3, 8), (3, 2, 8),
    (61, 60, 6), (62, 61, 9), (5, 6, 7), (6, 7, 7), (7, 5, 7), (0, 8, 2),
]


@pytest.fixture
def tiny_conn(tiny_atlas):
    import numpy as np

    from flybrain.connectome import Connectome

    pre, post, syn = (np.array(c, dtype=np.int32) for c in zip(*TINY_EDGES))
    return Connectome.from_edges(pre, post, syn, n_neurons=len(tiny_atlas.neurons), min_synapses=5)
