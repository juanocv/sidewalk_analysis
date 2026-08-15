import sidewalk_ai as sw


def test_package_import_keeps_heavy_exports_lazy():
    assert "StreetViewClient" in sw.__all__
    assert "build_depth" in sw.__all__
    assert "StreetViewClient" not in sw.__dict__


def test_model_factory_import_keeps_backends_lazy():
    from sidewalk_ai.models.factory import build_depth, build_segmenter

    assert callable(build_depth)
    assert callable(build_segmenter)
