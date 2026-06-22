import re
import sys

with open("pyproject.toml") as f:
    toml_content = f.read()

with open("scripts/bootstrapping/setup.ps1") as f:
    ps1_content = f.read()

toml_match = re.search(r'^version = "([^"]+)"', toml_content, re.MULTILINE)
ps1_match = re.search(r'^\$VOCALANCE_VERSION\s*=\s*[\'"]([^\'"]+)[\'"]', ps1_content, re.MULTILINE)

if not toml_match:
    print("ERROR: version not found in pyproject.toml")
    sys.exit(1)

if not ps1_match:
    print("ERROR: $VOCALANCE_VERSION not found in scripts/bootstrapping/setup.ps1")
    sys.exit(1)

toml_version = toml_match.group(1)
ps1_version = ps1_match.group(1)

if toml_version != ps1_version:
    print("ERROR: Version mismatch")
    print(f"  pyproject.toml:                  {toml_version}")
    print(f"  scripts/bootstrapping/setup.ps1: {ps1_version}")
    sys.exit(1)

print(f"OK: version {toml_version} matches in pyproject.toml and setup.ps1")
