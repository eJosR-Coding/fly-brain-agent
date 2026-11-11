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


# --- hardening (H1) ---------------------------------------------------------------

from flybrain.atlas import AtlasLoadError, UnknownNeuronError  # noqa: E402


def write_tsv(tmp_path, lines, name="bad.tsv"):
    path = tmp_path / name
    path.write_text("\n".join(lines) + "\n")
    return path


GOOD_ROW = "1\t1000\t2000\t100\tcentral\tCX\tPEN_a\tacetylcholine\tleft\tintrinsic"


def test_negative_limit_is_rejected(atlas):
    with pytest.raises(ValueError, match="limit must be >= 0"):
        atlas.find(limit=-1)


@pytest.mark.parametrize("bad", ["3", 2.5, True, [1]])
def test_non_int_limit_is_rejected(atlas, bad):
    with pytest.raises(TypeError, match="limit must be an int"):
        atlas.find(limit=bad)


def test_empty_and_whitespace_filters_mean_no_filter(atlas):
    assert len(atlas.find(cell_type="", nt="  ", super_class="\t", side="")) == 5


def test_filters_are_stripped(atlas):
    assert list(atlas.find(side="  Right ")["root_id"]) == [2, 4]


def test_non_string_filter_is_a_type_error(atlas):
    with pytest.raises(TypeError, match="nt must be a string"):
        atlas.find(nt=5)


def test_overlong_filter_is_rejected(atlas):
    with pytest.raises(ValueError, match="cell_type is too long"):
        atlas.find(cell_type="x" * 201)


def test_distinct_values_are_sorted_and_skip_missing(atlas):
    assert atlas.distinct("top_nt") == ["acetylcholine", "glutamate"]
    assert atlas.distinct("side") == ["left", "right"]


def test_describe_unknown_id_message_is_readable(atlas):
    with pytest.raises(UnknownNeuronError) as exc:
        atlas.describe(12)
    assert str(exc.value) == (
        "no neuron with root_id 12; get ids from search_neurons or list_selection"
    )
    assert isinstance(exc.value, KeyError)


@pytest.mark.parametrize("bad", [1.0, True, [1], None, "12a", "", "-3"])
def test_describe_rejects_non_int_ids(atlas, bad):
    with pytest.raises((TypeError, ValueError), match="root_id"):
        atlas.describe(bad)


def test_describe_accepts_all_digit_strings(atlas):
    assert atlas.describe(str(BIG_ID))["root_id"] == BIG_ID


def test_load_none_path_is_a_load_error():
    with pytest.raises(AtlasLoadError, match="no annotations path"):
        load_atlas(None)


def test_load_missing_file_is_a_load_error(tmp_path):
    with pytest.raises(AtlasLoadError, match="not found"):
        load_atlas(tmp_path / "nope.tsv")


def test_load_directory_is_a_load_error(tmp_path):
    with pytest.raises(AtlasLoadError, match="directory"):
        load_atlas(tmp_path)


def test_load_empty_file_is_a_load_error(tmp_path):
    path = tmp_path / "empty.tsv"
    path.write_text("")
    with pytest.raises(AtlasLoadError, match="empty"):
        load_atlas(path)


def test_load_header_only_means_no_neurons(tmp_path):
    with pytest.raises(AtlasLoadError, match="no neurons"):
        load_atlas(write_tsv(tmp_path, [HEADER]))


def test_load_names_only_the_missing_columns(tmp_path):
    header = HEADER.replace("\tflow", "").replace("pos_z", "pos_q")
    with pytest.raises(AtlasLoadError, match=r"missing required columns: pos_z, flow$"):
        load_atlas(write_tsv(tmp_path, [header, "1\t1\t1\t1\tc\tc\tc\tc\tc"]))


@pytest.mark.parametrize("pos", ["abc", "nan", "inf", ""])
def test_load_rejects_bad_positions(tmp_path, pos):
    row = GOOD_ROW.replace("\t1000\t", f"\t{pos}\t", 1)
    with pytest.raises(AtlasLoadError, match="positions"):
        load_atlas(write_tsv(tmp_path, [HEADER, row]))


@pytest.mark.parametrize("rid", ["abc", "99999999999999999999", "", "1.5"])
def test_load_rejects_bad_root_ids(tmp_path, rid):
    row = GOOD_ROW.replace("1\t", f"{rid}\t", 1)
    with pytest.raises(AtlasLoadError, match="root_id"):
        load_atlas(write_tsv(tmp_path, [HEADER, row]))


def test_load_rejects_duplicate_root_ids(tmp_path):
    with pytest.raises(AtlasLoadError, match="duplicate root_id"):
        load_atlas(write_tsv(tmp_path, [HEADER, GOOD_ROW, GOOD_ROW]))


def test_numeric_looking_cell_types_do_not_break_filters(tmp_path):
    row = "1\t1000\t2000\t100\tcentral\tCX\t123\tacetylcholine\tleft\tintrinsic"
    atlas = load_atlas(write_tsv(tmp_path, [HEADER, row]))
    assert len(atlas.find(cell_type="12")) == 1


def test_known_nt_is_loaded_as_its_first_real_transmitter(tmp_path):
    from conftest import HEADER, _tiny_rows
    from flybrain.atlas import load_atlas

    cases = ["acetylcholine; sNPF; acetylcholine, sNPF", "gaba-negative", "acetylcholine-negative, glutamate-negative",
             "histamine; acetylcholine, histamine", "gaba, nitric oxide", "sNPF", ""]
    rows = _tiny_rows()[:len(cases)]
    lines = [HEADER + "\tknown_nt"] + [f"{r}\t{c}" for r, c in zip(rows, cases)]
    path = tmp_path / "k.tsv"
    path.write_text("\n".join(lines) + "\n")
    got = load_atlas(path).neurons["known_nt"].tolist()
    assert got == ["acetylcholine", None, None, "histamine", "gaba", None, None]


def test_known_nt_is_optional_in_the_file(tiny_atlas):
    assert tiny_atlas.neurons["known_nt"].isna().all()
