"""Structured intake state and deterministic submission behavior."""

from pydantic import BaseModel, field_validator


class IntakeData(BaseModel):
    """User information collected during the intake conversation."""

    full_name: str | None = None
    phone_number: str | None = None
    email: str | None = None
    address: str | None = None
    preferred_contact_method: str | None = None

    @field_validator("*", mode="before")
    @classmethod
    def normalize_strings(cls, value: object) -> object:
        """Trim strings and normalize blank values to ``None``."""
        if isinstance(value, str):
            value = value.strip()
            return value or None
        return value

    def missing_fields(self) -> list[str]:
        """Return missing field names in intake order."""
        return [
            field_name
            for field_name in type(self).model_fields
            if getattr(self, field_name) is None
        ]

    def is_complete(self) -> bool:
        """Return whether every intake field has been provided."""
        return not self.missing_fields()


class SubmissionResult(BaseModel):
    """Result of an intake submission attempt."""

    success: bool
    message: str


def submit_intake(data: IntakeData) -> SubmissionResult:
    """Validate intake completeness without persisting or transmitting data."""
    missing = data.missing_fields()
    if missing:
        return SubmissionResult(
            success=False,
            message=f"Cannot submit intake; missing fields: {', '.join(missing)}.",
        )
    return SubmissionResult(
        success=True,
        message="Intake submitted successfully.",
    )
