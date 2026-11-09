import pytest

from flybrain.atlas import Atlas, load_atlas

BIG_ID = 720575940628857210

HEADER = "root_id\tpos_x\tpos_y\tpos_z\tsuper_class\tcell_class\tcell_type\ttop_nt\tside\tflow"
ROWS = [
    f"{BIG_ID}\t1000\t2000\t100\tcentral\tCX\tPEN_a\tacetylcholine\tleft\tintrinsic",
    "2\t3000\t2500\t200\tcentral\tCX\tPEN_b\tacetylcholine\tright\tintrinsic",
    "3\t500\t100\t50\toptic\tOL\tT4a\tacetylcholine\tleft\tafferent",
    "4\t800\t900\t400\tcentral\tCX\tEPG\tglutamate\tright\tintrinsic",
    "5\t200\t300\t10\tsensory\t\tORN_DA1\t\tleft\tafferent",
]


@pytest.fixture
def atlas(tmp_path) -> Atlas:
    path = tmp_path / "ann.tsv"
    path.write_text("\n".join([HEADER, *ROWS]) + "\n")
    return load_atlas(path)


def test_large_root_ids_survive_as_int64(atlas):
    assert BIG_ID in set(atlas.neurons["root_id"])
    assert str(atlas.neurons["root_id"].dtype) == "int64"


def test_z_is_ten_times_thicker_than_x_and_y(atlas):
    row = atlas.neurons[atlas.neurons["root_id"] == BIG_ID].iloc[0]
    assert (row.x_um, row.y_um, row.z_um) == (4.0, 8.0, 4.0)


def test_cell_type_is_a_case_insensitive_prefix(atlas):
    assert set(atlas.find(cell_type="pen")["cell_type"]) == {"PEN_a", "PEN_b"}


def test_filters_are_anded(atlas):
    hits = atlas.find(cell_type="PEN", side="Right")
    assert list(hits["cell_type"]) == ["PEN_b"]


def test_nt_match_is_exact_and_case_insensitive(atlas):
    assert len(atlas.find(nt="GLUTAMATE")) == 1
    assert len(atlas.find(nt="gluta")) == 0


def test_missing_values_never_match_a_filter(atlas):
    assert 5 not in set(atlas.find(nt="acetylcholine")["root_id"])
    assert 5 not in set(atlas.find(super_class="central")["root_id"])


def test_limit_caps_results(atlas):
    assert len(atlas.find(limit=2)) == 2


def test_no_filters_returns_everything(atlas):
    assert len(atlas.find()) == 5


def test_count_by_sorts_descending_and_labels_missing_as_unknown(atlas):
    counts = atlas.count_by("top_nt")
    assert counts == {"acetylcholine": 3, "glutamate": 1, "unknown": 1}
    assert list(counts)[0] == "acetylcholine"


def test_count_by_respects_filters(atlas):
    assert atlas.count_by("side", super_class="central") == {"right": 2, "left": 1}


def test_count_by_unknown_field_names_allowed_fields(atlas):
    with pytest.raises(ValueError, match="top_nt"):
        atlas.count_by("color")


def test_describe_returns_plain_json_types(atlas):
    info = atlas.describe(BIG_ID)
    assert info["root_id"] == BIG_ID and type(info["root_id"]) is int
    assert type(info["x_um"]) is float
    assert info["cell_type"] == "PEN_a"


def test_describe_missing_text_is_none(atlas):
    assert atlas.describe(5)["top_nt"] is None


def test_describe_unknown_id_raises_key_error(atlas):
    with pytest.raises(KeyError):
        atlas.describe(999)


def test_extent_is_max_minus_min_per_axis(atlas):
    # x: 3000*4 - 200*4 = 11.2 um, y: 2500*4 - 100*4 = 9.6, z: 400*40 - 10*40 = 15.6
    assert atlas.extent_um() == pytest.approx((11.2, 9.6, 15.6))


def test_real_data_has_full_brain():
    from pathlib import Path

    path = Path("data/raw/flywire_neuron_annotations.tsv")
    if not path.exists():
        pytest.skip("real annotations not present")
    real = load_atlas(path)
    assert len(real.neurons) == 139_248
    assert real.extent_um() == pytest.approx((815, 392, 278), rel=0.02)
