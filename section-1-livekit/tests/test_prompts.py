"""Tests for the structured intake system prompt."""

from prompts import SYSTEM_PROMPT


REQUIRED_FIELDS = {
    "full_name",
    "phone_number",
    "email",
    "address",
    "preferred_contact_method",
}


def test_system_prompt_is_non_empty_string() -> None:
    assert isinstance(SYSTEM_PROMPT, str)
    assert SYSTEM_PROMPT.strip()


def test_system_prompt_describes_arabic_and_english_behavior() -> None:
    prompt = SYSTEM_PROMPT.lower()

    assert "arabic" in prompt
    assert "english" in prompt
    assert "language the user primarily uses" in prompt


def test_system_prompt_includes_every_required_field() -> None:
    for field in REQUIRED_FIELDS:
        assert field in SYSTEM_PROMPT


def test_system_prompt_requires_explicit_confirmation() -> None:
    prompt = SYSTEM_PROMPT.lower()

    assert "explicit confirmation" in prompt
    assert prompt.index("explicit confirmation") < prompt.index(
        "call the appropriate available function"
    )


def test_system_prompt_forbids_inventing_missing_data() -> None:
    prompt = SYSTEM_PROMPT.lower()

    assert "never invent" in prompt
    assert "never invent, assume" in prompt

def test_prompt_protects_arabic_and_english_personal_names() -> None:
    prompt = SYSTEM_PROMPT.lower()

    assert "personal names" in prompt
    assert "never translate" in prompt
    assert "never translate, transliterate" in prompt
    assert "explicitly confirm that name" in prompt
    assert "below 0.75" in prompt
    assert "repeat or spell" in prompt
    assert "fully replaces the previous name" in prompt


def test_prompt_switches_from_english_to_arabic() -> None:
    prompt = SYSTEM_PROMPT.lower()

    assert "english to arabic" in prompt
    assert "switch immediately" in prompt
    assert "language the user is currently using" in prompt
