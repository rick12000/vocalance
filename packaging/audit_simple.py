import logging
from pathlib import Path
import sys
from bundle_exclusions import should_exclude_uncovered_dll

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Track which file extensions to audit
AUDIT_EXTENSIONS = {'.dll'}


def get_site_packages_dlls():
    """Extract all DLLs from site-packages directory."""
    # Get site-packages path from current Python environment
    site_packages = Path(sys.base_prefix) / 'Lib' / 'site-packages'
    
    if not site_packages.exists():
        logging.error(f"site-packages not found: {site_packages}")
        return set()
    
    dlls = set()
    for file_path in site_packages.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in AUDIT_EXTENSIONS:
            # Store just the filename (not full path) for cross-reference
            dlls.add(file_path.name.lower())
    
    logging.info(f"Found {len(dlls)} DLLs in site-packages")
    return dlls


def get_dist_dlls(dist_path):
    """Extract all DLLs from PyInstaller distribution."""
    if not dist_path.exists():
        logging.error(f"Distribution path not found: {dist_path}")
        return []
    
    dlls = []
    for file_path in dist_path.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in AUDIT_EXTENSIONS:
            # Store relative path from _internal for reporting
            rel_path = file_path.relative_to(dist_path)
            dlls.append({
                'name': file_path.name.lower(),
                'path': str(rel_path),
                'full_path': file_path
            })
    
    logging.info(f"Found {len(dlls)} DLLs in distribution")
    return dlls


def categorize_uncovered_dlls(uncovered_dlls):
    """Separate uncovered DLLs into proprietary (exclude) and ambiguous (review).
    
    Args:
        uncovered_dlls: List of uncovered DLL info dicts
    
    Returns:
        Tuple of (proprietary_list, ambiguous_list)
    """
    proprietary = []
    ambiguous = []
    
    for dll_info in uncovered_dlls:
        dll_name = dll_info['name']
        if should_exclude_uncovered_dll(dll_name):
            proprietary.append(dll_info)
        else:
            ambiguous.append(dll_info)
    
    return proprietary, ambiguous


def main():
    project_root = Path(__file__).parent.parent
    dist_path = project_root / 'dist' / 'vocalance' / '_internal'
    audit_reports_dir = project_root / 'packaging' / 'audit_reports'
    audit_reports_dir.mkdir(exist_ok=True)
    output_covered = audit_reports_dir / 'simple_audit_covered.txt'
    output_proprietary = audit_reports_dir / 'simple_audit_proprietary.txt'
    output_ambiguous = audit_reports_dir / 'simple_audit_ambiguous.txt'
    output_summary = audit_reports_dir / 'simple_audit_summary.txt'
    
    logging.info(f"Audit Extensions: {AUDIT_EXTENSIONS}")
    logging.info(f"Distribution path: {dist_path}")
    
    # Step 1: Get all DLLs in site-packages
    site_dlls = get_site_packages_dlls()
    
    # Step 2: Get all DLLs in distribution
    dist_dlls = get_dist_dlls(dist_path)
    
    # Step 3: Cross-reference
    covered = []
    uncovered = []
    
    for dll_info in dist_dlls:
        dll_name = dll_info['name']
        if dll_name in site_dlls:
            covered.append(dll_info)
        else:
            uncovered.append(dll_info)
    
    # Step 4: Categorize uncovered DLLs
    proprietary, ambiguous = categorize_uncovered_dlls(uncovered)
    
    logging.info(f"\nResults:")
    logging.info(f"  Covered (in site-packages): {len(covered)}")
    logging.info(f"  Uncovered (NOT in site-packages): {len(uncovered)}")
    logging.info(f"    → Proprietary/System (auto-exclude): {len(proprietary)}")
    logging.info(f"    → Ambiguous (requires review): {len(ambiguous)}")
    logging.info(f"  Total DLLs in distribution: {len(dist_dlls)}")
    
    # Write covered list
    with open(output_covered, 'w', encoding='utf-8') as f:
        f.write('\n'.join(sorted([dll['path'] for dll in covered])))
    logging.info(f"\nCovered list written to: {output_covered}")
    
    # Write proprietary list
    with open(output_proprietary, 'w', encoding='utf-8') as f:
        f.write('\n'.join(sorted([dll['path'] for dll in proprietary])))
    logging.info(f"Proprietary list written to: {output_proprietary}")
    
    # Write ambiguous list
    with open(output_ambiguous, 'w', encoding='utf-8') as f:
        f.write('\n'.join(sorted([dll['path'] for dll in ambiguous])))
    logging.info(f"Ambiguous list written to: {output_ambiguous}")
    
    # Write summary report
    with open(output_summary, 'w', encoding='utf-8') as f:
        f.write("=" * 100 + "\n")
        f.write("SIMPLE DLL COMPLIANCE AUDIT - LAYER 1\n")
        f.write("=" * 100 + "\n\n")
        
        f.write("Methodology:\n")
        f.write("1. Extract all DLLs from site-packages (assumed covered by package licenses)\n")
        f.write("2. Extract all DLLs from PyInstaller distribution\n")
        f.write("3. Cross-reference: DLLs in dist but NOT in site-packages are uncovered\n")
        f.write("4. Categorize uncovered: proprietary/system vs. ambiguous\n\n")
        
        f.write("=" * 100 + "\n")
        f.write("SUMMARY\n")
        f.write("=" * 100 + "\n")
        f.write(f"Total DLLs in distribution: {len(dist_dlls)}\n")
        f.write(f"Covered by site-packages: {len(covered)}\n")
        f.write(f"Uncovered (require investigation): {len(uncovered)}\n")
        f.write(f"  → Proprietary/System (auto-exclude): {len(proprietary)}\n")
        f.write(f"  → Ambiguous (manual review): {len(ambiguous)}\n\n")
        
        f.write("=" * 100 + "\n")
        f.write(f"PROPRIETARY/SYSTEM DLLs - AUTO EXCLUDE ({len(proprietary)})\n")
        f.write("=" * 100 + "\n\n")
        f.write("These should be excluded from the bundle:\n\n")
        
        if proprietary:
            for dll in sorted(proprietary, key=lambda x: x['path']):
                f.write(f"{dll['path']}\n")
        else:
            f.write("None\n")
        
        f.write("\n" + "=" * 100 + "\n")
        f.write(f"AMBIGUOUS DLLs - REQUIRES MANUAL REVIEW ({len(ambiguous)})\n")
        f.write("=" * 100 + "\n\n")
        f.write("These DLLs are not in site-packages but don't match proprietary patterns.\n")
        f.write("Review each to determine if it should be bundled:\n\n")
        
        if ambiguous:
            for dll in sorted(ambiguous, key=lambda x: x['path']):
                f.write(f"{dll['path']}\n")
        else:
            f.write("None - all uncovered DLLs are proprietary!\n")
        
        f.write("\n" + "=" * 100 + "\n")
        f.write(f"COVERED DLLs BY PACKAGE ({len(covered)})\n")
        f.write("=" * 100 + "\n\n")
        
        if covered:
            for dll in sorted(covered, key=lambda x: x['path'])[:50]:
                f.write(f"{dll['path']}\n")
            if len(covered) > 50:
                f.write(f"\n... and {len(covered) - 50} more covered DLLs\n")
        else:
            f.write("None\n")
    
    logging.info(f"Summary report written to: {output_summary}")


if __name__ == '__main__':
    main()
