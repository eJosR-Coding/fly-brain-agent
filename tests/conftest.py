import pytest

from flybrain.atlas import Atlas, load_atlas

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
