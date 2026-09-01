from datetime import date

from examples.epa_2023.download_inputs import input_objects


def test_input_objects_keep_scenario_and_meteorology_years_explicit() -> None:
    objects = input_objects(date(2016, 7, 15))

    assert len(objects) == 4
    assert "METCRO3D.12US1.35L.160715" in objects
    emissions = next(name for name in objects if name.startswith("emis_"))
    assert "20160715" in emissions
    assert "2023gf" in emissions
    assert objects[emissions].startswith("s3://2016v3platform/")
