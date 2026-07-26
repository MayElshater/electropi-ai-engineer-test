"""Tests for deterministic structured intake tool behavior."""

from intake import IntakeData
from tools import process_structured_intake


COMPLETE_DATA = {
    "full_name": "Ada Lovelace",
    "phone_number": "+20 100 123 4567",
    "email": "ada@example.com",
    "address": "1 Computing Way",
    "preferred_contact_method": "email",
}
RESULT_KEYS = {
    "success",
    "message",
    "missing_fields",
    "requires_confirmation",
}


def test_partial_updates_mutate_state_and_preserve_existing_values() -> None:
    intake = IntakeData(
        full_name="Ada Lovelace",
        phone_number="+20 100 123 4567",
    )

    process_structured_intake(intake, email="ada@example.com")

    assert intake.full_name == "Ada Lovelace"
    assert intake.phone_number == "+20 100 123 4567"
    assert intake.email == "ada@example.com"


def test_updates_reuse_intake_normalization() -> None:
    intake = IntakeData()

    process_structured_intake(
        intake,
        full_name="  Ada Lovelace  ",
        email=" \t ",
    )

    assert intake.full_name == "Ada Lovelace"
    assert intake.email is None


def test_supplied_intake_object_is_preserved() -> None:
    intake = IntakeData()
    identity = id(intake)

    process_structured_intake(intake, full_name="Ada Lovelace")

    assert id(intake) == identity
    assert intake.full_name == "Ada Lovelace"


def test_unconfirmed_intake_is_not_submitted() -> None:
    result = process_structured_intake(
        IntakeData(**COMPLETE_DATA),
        confirmed=False,
    )

    assert result["success"] is False
    assert result["requires_confirmation"] is True
    assert "confirmation" in result["message"].lower()


def test_confirmed_incomplete_intake_is_rejected() -> None:
    result = process_structured_intake(
        IntakeData(full_name="Ada Lovelace"),
        confirmed=True,
    )

    assert result["success"] is False
    assert result["requires_confirmation"] is False
    assert "phone_number" in result["message"]


def test_confirmed_complete_intake_is_accepted() -> None:
    result = process_structured_intake(
        IntakeData(**COMPLETE_DATA),
        confirmed=True,
    )

    assert result["success"] is True
    assert result["missing_fields"] == []
    assert result["requires_confirmation"] is False


def test_missing_fields_use_expected_order() -> None:
    result = process_structured_intake(
        IntakeData(phone_number="+20 100 123 4567", address="Cairo")
    )

    assert result["missing_fields"] == [
        "full_name",
        "email",
        "preferred_contact_method",
    ]


def test_results_have_exact_keys() -> None:
    intake = IntakeData()

    unconfirmed = process_structured_intake(intake)
    confirmed = process_structured_intake(intake, confirmed=True)

    assert set(unconfirmed) == RESULT_KEYS
    assert set(confirmed) == RESULT_KEYS


def test_failure_message_does_not_expose_sensitive_values() -> None:
    sensitive_name = "Private Person"
    sensitive_phone = "+20 111 987 6543"

    result = process_structured_intake(
        IntakeData(),
        full_name=sensitive_name,
        phone_number=sensitive_phone,
        confirmed=True,
    )

    assert result["success"] is False
    assert sensitive_name not in result["message"]
    assert sensitive_phone not in result["message"]
