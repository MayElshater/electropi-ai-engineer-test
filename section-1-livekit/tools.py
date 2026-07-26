"""Deterministic tools for updating and submitting structured intake data."""

from typing import TypedDict

from intake import IntakeData, submit_intake


class ToolResult(TypedDict):
    """LLM-readable result returned by the intake submission tool."""

    success: bool
    message: str
    missing_fields: list[str]
    requires_confirmation: bool


def process_structured_intake(
    intake_data: IntakeData,
    full_name: str | None = None,
    phone_number: str | None = None,
    email: str | None = None,
    address: str | None = None,
    preferred_contact_method: str | None = None,
    confirmed: bool = False,
) -> ToolResult:
    """Update supplied fields and submit only after explicit confirmation."""
    supplied = {
        "full_name": full_name,
        "phone_number": phone_number,
        "email": email,
        "address": address,
        "preferred_contact_method": preferred_contact_method,
    }
    updates = {name: value for name, value in supplied.items() if value is not None}

    if updates:
        candidate = IntakeData(**(intake_data.model_dump() | updates))
        for name in updates:
            setattr(intake_data, name, getattr(candidate, name))

    missing = intake_data.missing_fields()
    if not confirmed:
        return {
            "success": False,
            "message": "Explicit confirmation is required before submission.",
            "missing_fields": missing,
            "requires_confirmation": True,
        }

    submission = submit_intake(intake_data)
    return {
        "success": submission.success,
        "message": submission.message,
        "missing_fields": missing,
        "requires_confirmation": False,
    }
