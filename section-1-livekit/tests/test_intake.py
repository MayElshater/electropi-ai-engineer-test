"""Tests for structured intake state and submission."""

from intake import IntakeData, submit_intake


COMPLETE_DATA = {
    "full_name": "Ada Lovelace",
    "phone_number": "+20 100 123 4567",
    "email": "ada@example.com",
    "address": "1 Computing Way",
    "preferred_contact_method": "email",
}


def test_default_intake_is_incomplete() -> None:
    data = IntakeData()

    assert not data.is_complete()


def test_whitespace_is_trimmed() -> None:
    data = IntakeData(full_name="  Ada Lovelace  ", email="\tada@example.com\n")

    assert data.full_name == "Ada Lovelace"
    assert data.email == "ada@example.com"


def test_blank_strings_become_none() -> None:
    data = IntakeData(full_name="", phone_number=" \t\n ")

    assert data.full_name is None
    assert data.phone_number is None


def test_missing_fields_preserves_required_order() -> None:
    data = IntakeData(phone_number="+20 100 123 4567", address="Cairo")

    assert data.missing_fields() == [
        "full_name",
        "email",
        "preferred_contact_method",
    ]


def test_complete_intake_is_complete() -> None:
    data = IntakeData(**COMPLETE_DATA)

    assert data.is_complete()
    assert data.missing_fields() == []


def test_submit_intake_rejects_incomplete_data() -> None:
    result = submit_intake(IntakeData(full_name="Ada Lovelace"))

    assert result.success is False
    assert "phone_number" in result.message


def test_submit_intake_accepts_complete_data() -> None:
    result = submit_intake(IntakeData(**COMPLETE_DATA))

    assert result.success is True
    assert result.message == "Intake submitted successfully."


def test_failure_message_lists_missing_fields_without_sensitive_values() -> None:
    sensitive_name = "Private Person"
    sensitive_phone = "+20 111 987 6543"
    data = IntakeData(full_name=sensitive_name, phone_number=sensitive_phone)

    result = submit_intake(data)

    assert result.success is False
    assert "email" in result.message
    assert "address" in result.message
    assert "preferred_contact_method" in result.message
    assert sensitive_name not in result.message
    assert sensitive_phone not in result.message
