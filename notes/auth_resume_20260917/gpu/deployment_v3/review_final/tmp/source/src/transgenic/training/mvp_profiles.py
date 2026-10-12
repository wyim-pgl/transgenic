"""Lightweight seed identities, separate from the fixed MVP runtime recipe."""

LEGACY_PROFILE = "historical-three-seed"
DEVELOPMENT_PROFILE = "development-single"


def validate_seed_profile(profile, seed) -> None:
    """Reject ambiguous/cross-profile identities; callers default omitted legacy fields."""
    if type(profile) is not str or profile not in (LEGACY_PROFILE, DEVELOPMENT_PROFILE):
        raise ValueError("unknown MVP seed profile")
    seeds = (1, 2, 3) if profile == LEGACY_PROFILE else (123,)
    if type(seed) is not int or seed not in seeds:
        raise ValueError(f"MVP profile {profile} requires exact integer seed in {seeds}")
