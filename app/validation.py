def validate_required_text(
    value: str,
    field_name: str,
    *,
    max_length: int | None = None,
) -> str:
    """Validate and trim a required string value."""

    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")

    cleaned_value = value.strip()

    if not cleaned_value:
        raise ValueError(f"{field_name} is required.")

    if max_length is not None and len(cleaned_value) > max_length:
        raise ValueError(
            f"{field_name} cannot exceed {max_length:,} characters."
        )

    return cleaned_value
