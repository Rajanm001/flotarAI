import pytest

from app.services.features import (
    MalformedDobError,
    age_from_dob,
    load_users_from_csv,
)


def test_age_from_dob_raises_on_malformed_input():
    with pytest.raises(MalformedDobError):
        age_from_dob("not-a-date")


def test_age_from_dob_computes_correct_year_difference():
    import datetime as dt
    age = age_from_dob("2000-01-01", reference_date=dt.date(2026, 8, 24))
    assert age == 26.0


def test_load_users_from_csv_raises_on_duplicate_user_id(tmp_path):
    csv_path = tmp_path / "dupes.csv"
    csv_path.write_text(
        "UserID,Name,Gender,DOB,Interests,City,Country\n"
        "1,Alice,Female,1990-01-01,'Music',Delhi,India\n"
        "1,Bob,Male,1991-01-01,'Sports',Mumbai,India\n"
    )
    with pytest.raises(ValueError, match="duplicate UserID"):
        load_users_from_csv(csv_path)


def test_load_users_from_csv_succeeds_with_unique_ids(tmp_path):
    csv_path = tmp_path / "clean.csv"
    csv_path.write_text(
        "UserID,Name,Gender,DOB,Interests,City,Country\n"
        "1,Alice,Female,1990-01-01,'Music',Delhi,India\n"
        "2,Bob,Male,1991-01-01,'Sports',Mumbai,India\n"
    )
    users = load_users_from_csv(csv_path)
    assert len(users) == 2
    assert {u.user_id for u in users} == {1, 2}
