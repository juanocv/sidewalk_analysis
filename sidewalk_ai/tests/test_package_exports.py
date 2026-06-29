import sidewalk_ai as sw


def test_package_import_keeps_heavy_exports_lazy():
    assert "StreetViewClient" in sw.__all__
    assert "build_depth" in sw.__all__
    assert "StreetViewClient" not in sw.__dict__
