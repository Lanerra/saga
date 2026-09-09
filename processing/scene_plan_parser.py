"""Pure scene-plan parsing shared by context consumers."""


def extract_scene_characters(scene: dict) -> list[str]:
    """Return stripped, nonempty character names in first-occurrence order."""
    seen: set[str] = set()
    characters: list[str] = []
    for name in scene.get("characters", []):
        stripped = name.strip()
        if stripped and stripped not in seen:
            seen.add(stripped)
            characters.append(stripped)
    return characters
