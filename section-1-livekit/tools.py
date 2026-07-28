"""Deterministic tools for updating and submitting structured intake data."""

from dataclasses import dataclass
from typing import TypedDict

from intake import IntakeData, submit_intake

NAME_CONFIDENCE_THRESHOLD = 0.75


class ToolResult(TypedDict):
    """LLM-readable result returned by the intake submission tool."""

    success: bool
    message: str
    missing_fields: list[str]
    requires_confirmation: bool


@dataclass
class NameCaptureState:
    """Pending name candidate awaiting explicit user confirmation."""

    value: str | None = None
    confidence: float | None = None


def _result(
    intake_data: IntakeData,
    message: str,
    *,
    requires_confirmation: bool,
) -> ToolResult:
    """Return the stable tool-result shape without personal values."""
    return {
        "success": False,
        "message": message,
        "missing_fields": intake_data.missing_fields(),
        "requires_confirmation": requires_confirmation,
    }


def process_structured_intake(
    intake_data: IntakeData,
    full_name: str | None = None,
    phone_number: str | None = None,
    email: str | None = None,
    address: str | None = None,
    preferred_contact_method: str | None = None,
    confirmed: bool = False,
    *,
    full_name_confirmed: bool = False,
    full_name_confidence: float | None = None,
    name_state: NameCaptureState | None = None,
) -> ToolResult:
    """Update supplied fields while gating names on confidence and confirmation."""
    pending_name = name_state if name_state is not None else NameCaptureState()
    other_updates = {
        name: value
        for name, value in {
            "phone_number": phone_number,
            "email": email,
            "address": address,
            "preferred_contact_method": preferred_contact_method,
        }.items()
        if value is not None
    }

    if other_updates:
        candidate = IntakeData(**(intake_data.model_dump() | other_updates))
        for name in other_updates:
            setattr(intake_data, name, getattr(candidate, name))

    if full_name is not None:
        pending_name.value = IntakeData(full_name=full_name).full_name
        pending_name.confidence = full_name_confidence
        if (
            pending_name.confidence is not None
            and pending_name.confidence < NAME_CONFIDENCE_THRESHOLD
        ):
            return _result(
                intake_data,
                "The full-name transcript confidence is low; ask the user to repeat or spell the name before saving it.",
                requires_confirmation=True,
            )
        return _result(
            intake_data,
            "Explicit confirmation of the captured full name is required before saving it.",
            requires_confirmation=True,
        )

    if full_name_confirmed:
        if pending_name.value is None:
            return _result(
                intake_data,
                "A non-blank full name is required before name confirmation.",
                requires_confirmation=True,
            )
        if (
            pending_name.confidence is not None
            and pending_name.confidence < NAME_CONFIDENCE_THRESHOLD
        ):
            return _result(
                intake_data,
                "The full-name transcript confidence is low; ask the user to repeat or spell the name before saving it.",
                requires_confirmation=True,
            )

        intake_data.full_name = pending_name.value
        pending_name.value = None
        pending_name.confidence = None

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
