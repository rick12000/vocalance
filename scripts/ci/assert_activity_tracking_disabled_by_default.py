"""CI guard: ActivityTrackingConfig.enabled must default to False (privacy)."""
import ast
import sys
from pathlib import Path


def main():
    config_path = Path("vocalance/app/config/app_config.py")
    if not config_path.exists():
        print(f"ERROR: {config_path} not found.")
        return 1

    tree = ast.parse(config_path.read_text(encoding="utf-8"))

    try:
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "ActivityTrackingConfig")
        field = next(n for n in cls.body if isinstance(n, ast.AnnAssign) and getattr(n.target, "id", "") == "enabled")
    except StopIteration:
        print("ERROR: ActivityTrackingConfig.enabled definition not found.")
        return 1

    is_false = False
    # Handle `enabled: bool = False`
    if isinstance(field.value, ast.Constant) and field.value.value is False:
        is_false = True
    # Handle `enabled: bool = Field(default=False, ...)`
    elif isinstance(field.value, ast.Call):
        is_false = any(
            kw.arg == "default" and isinstance(kw.value, ast.Constant) and kw.value.value is False for kw in field.value.keywords
        )

    if not is_false:
        print("ERROR: ActivityTrackingConfig.enabled MUST default to False for privacy.")
        return 1

    print("Check passed: activity tracking enabled defaults to False.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
