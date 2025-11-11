from pydantic import BaseModel

class ProprietaryDllPatterns(BaseModel):
    """Patterns for proprietary/system DLLs that should not be bundled.
    
    Only applied to UNCOVERED DLLs to avoid excluding legitimate package DLLs.
    All patterns are permissive licenses verified against authoritative sources.
    """
    
    windows_api_forwarders: list[str] = [
        "api-ms-win-core-",
        "api-ms-win-crt-",
    ]
    visual_cpp_runtime: list[str] = [
        "vcruntime",
        "msvcp140",
        "ucrtbase",
    ]
    microsoft_mfc: list[str] = [
        "mfc140u.dll",
    ]

    @property
    def all_patterns(self) -> list[str]:
        """Flatten all exclusion patterns into single list."""
        patterns = []
        patterns.extend(self.windows_api_forwarders)
        patterns.extend(self.visual_cpp_runtime)
        patterns.extend(self.microsoft_mfc)
        return patterns

    @property
    def exclusion_rationale(self) -> dict[str, str]:
        """Document why each category is excluded."""
        return {
            "windows_api_forwarders": "Windows system DLLs - provided by OS, not redistributable as individual files",
            "visual_cpp_runtime": "Visual C++ Runtime - handled by vc_redist.exe installer",
        }


PROPRIETARY_PATTERNS = ProprietaryDllPatterns()


def should_exclude_uncovered_dll(dll_name: str) -> bool:
    """Determine if an uncovered DLL matches exclusion patterns.
    
    IMPORTANT: Only call this on DLLs that are NOT in site-packages.
    This ensures open-source packages with similar names won't be excluded.
    
    Args:
        dll_name: Filename of the DLL (e.g., 'vcruntime140.dll')
    
    Returns:
        True if the DLL should be excluded from bundling.
    """
    dll_lower = dll_name.lower()
    return any(pattern.lower() in dll_lower for pattern in PROPRIETARY_PATTERNS.all_patterns)
