"""Tests for deterministic structured intake tool behavior."""

from intake import IntakeData
from tools import NameCaptureState, process_structured_intake


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
    intake = IntakeData(full_name="Ada Lovelace", phone_number="+20 100 123 4567")

    process_structured_intake(intake, email="ada@example.com")

    assert intake.full_name == "Ada Lovelace"
    assert intake.phone_number == "+20 100 123 4567"
    assert intake.email == "ada@example.com"


def test_non_name_updates_reuse_intake_normalization() -> None:
    intake = IntakeData()

    process_structured_intake(intake, email=" \t ")

    assert intake.email is None


def test_arabic_name_requires_confirmation_before_state_update() -> None:
    intake = IntakeData()
    name_state = NameCaptureState()

    captured = process_structured_intake(
        intake,
        full_name="مي محمد",
        full_name_confidence=0.92,
        name_state=name_state,
    )

    assert intake.full_name is None
    assert captured["requires_confirmation"] is True
    assert "name" in captured["message"].lower()

    process_structured_intake(
        intake,
        full_name_confirmed=True,
        name_state=name_state,
    )

    assert intake.full_name == "مي محمد"


def test_name_cannot_be_captured_and_confirmed_in_same_call() -> None:
    intake = IntakeData()
    name_state = NameCaptureState()

    result = process_structured_intake(
        intake,
        full_name="مي محمد",
        full_name_confidence=0.95,
        full_name_confirmed=True,
        name_state=name_state,
    )

    assert intake.full_name is None
    assert result["requires_confirmation"] is True
    assert "confirmation" in result["message"].lower()
def test_low_confidence_name_must_be_repeated_or_spelled() -> None:
    intake = IntakeData()
    name_state = NameCaptureState()

    result = process_structured_intake(
        intake,
        full_name="Es Mimahy Mohammad",
        full_name_confidence=0.589,
        full_name_confirmed=True,
        name_state=name_state,
    )

    assert intake.full_name is None
    assert result["requires_confirmation"] is True
    assert "repeat or spell" in result["message"].lower()
    assert "Es Mimahy Mohammad" not in result["message"]


def test_confirmed_name_correction_fully_replaces_previous_value() -> None:
    intake = IntakeData(full_name="May Mohamed")
    name_state = NameCaptureState()

    process_structured_intake(
        intake,
        full_name="مي محمد",
        full_name_confidence=0.95,
        name_state=name_state,
    )
    assert intake.full_name == "May Mohamed"

    process_structured_intake(
        intake,
        full_name_confirmed=True,
        name_state=name_state,
    )

    assert intake.full_name == "مي محمد"


def test_supplied_intake_object_is_preserved() -> None:
    intake = IntakeData()
    identity = id(intake)
    name_state = NameCaptureState()

    process_structured_intake(
        intake,
        full_name="Ada Lovelace",
        full_name_confidence=0.9,
        name_state=name_state,
    )
    process_structured_intake(
        intake,
        full_name_confirmed=True,
        name_state=name_state,
    )

    assert id(intake) == identity
    assert intake.full_name == "Ada Lovelace"


def test_unconfirmed_intake_is_not_submitted() -> None:
    result = process_structured_intake(IntakeData(**COMPLETE_DATA), confirmed=False)

    assert result["success"] is False
    assert result["requires_confirmation"] is True
    assert "confirmation" in result["message"].lower()


def test_confirmed_incomplete_intake_is_rejected() -> None:
    result = process_structured_intake(IntakeData(full_name="Ada Lovelace"), confirmed=True)

    assert result["success"] is False
    assert result["requires_confirmation"] is False
    assert "phone_number" in result["message"]


def test_confirmed_complete_intake_is_accepted() -> None:
    result = process_structured_intake(IntakeData(**COMPLETE_DATA), confirmed=True)

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
        full_name_confidence=0.5,
        phone_number=sensitive_phone,
        confirmed=True,
    )

    assert result["success"] is False
    assert sensitive_name not in result["message"]
    assert sensitive_phone not in result["message"]
