#!/usr/bin/env python3
"""
5G Emulator API - Static Analysis Code Quality Checker

This script performs static analysis to catch issues BEFORE deployment:
1. Import Validation - Check all imports are valid
2. Dependency Validation - Check imports match requirements.txt
3. Health Endpoint Validation - Verify /health endpoints exist with required fields
4. Port Configuration Validation - Check for port conflicts
5. Main Block Validation - Verify proper __main__ blocks with argparse

Usage:
    python scripts/check_code_quality.py

Exit Codes:
    0 - All checks passed
    1 - One or more checks failed
"""

import ast
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass, field


# Standard library modules (Python 3.x built-ins)
STDLIB_MODULES = {
    'abc', 'aifc', 'argparse', 'array', 'ast', 'asynchat', 'asyncio', 'asyncore',
    'atexit', 'audioop', 'base64', 'bdb', 'binascii', 'binhex', 'bisect',
    'builtins', 'bz2', 'calendar', 'cgi', 'cgitb', 'chunk', 'cmath', 'cmd',
    'code', 'codecs', 'codeop', 'collections', 'colorsys', 'compileall',
    'concurrent', 'configparser', 'contextlib', 'contextvars', 'copy', 'copyreg',
    'cProfile', 'crypt', 'csv', 'ctypes', 'curses', 'dataclasses', 'datetime',
    'dbm', 'decimal', 'difflib', 'dis', 'distutils', 'doctest', 'email',
    'encodings', 'enum', 'errno', 'faulthandler', 'fcntl', 'filecmp', 'fileinput',
    'fnmatch', 'fractions', 'ftplib', 'functools', 'gc', 'getopt', 'getpass',
    'gettext', 'glob', 'graphlib', 'grp', 'gzip', 'hashlib', 'heapq', 'hmac',
    'html', 'http', 'idlelib', 'imaplib', 'imghdr', 'imp', 'importlib', 'inspect',
    'io', 'ipaddress', 'itertools', 'json', 'keyword', 'lib2to3', 'linecache',
    'locale', 'logging', 'lzma', 'mailbox', 'mailcap', 'marshal', 'math',
    'mimetypes', 'mmap', 'modulefinder', 'multiprocessing', 'netrc', 'nis',
    'nntplib', 'numbers', 'operator', 'optparse', 'os', 'ossaudiodev', 'parser',
    'pathlib', 'pdb', 'pickle', 'pickletools', 'pipes', 'pkgutil', 'platform',
    'plistlib', 'poplib', 'posix', 'posixpath', 'pprint', 'profile', 'pstats',
    'pty', 'pwd', 'py_compile', 'pyclbr', 'pydoc', 'queue', 'quopri', 'random',
    're', 'readline', 'reprlib', 'resource', 'rlcompleter', 'runpy', 'sched',
    'secrets', 'select', 'selectors', 'shelve', 'shlex', 'shutil', 'signal',
    'site', 'smtpd', 'smtplib', 'sndhdr', 'socket', 'socketserver', 'spwd',
    'sqlite3', 'ssl', 'stat', 'statistics', 'string', 'stringprep', 'struct',
    'subprocess', 'sunau', 'symbol', 'symtable', 'sys', 'sysconfig', 'syslog',
    'tabnanny', 'tarfile', 'telnetlib', 'tempfile', 'termios', 'test', 'textwrap',
    'threading', 'time', 'timeit', 'tkinter', 'token', 'tokenize', 'trace',
    'traceback', 'tracemalloc', 'tty', 'turtle', 'turtledemo', 'types', 'typing',
    'unicodedata', 'unittest', 'urllib', 'uu', 'uuid', 'venv', 'warnings', 'wave',
    'weakref', 'webbrowser', 'winreg', 'winsound', 'wsgiref', 'xdrlib', 'xml',
    'xmlrpc', 'zipapp', 'zipfile', 'zipimport', 'zlib', '_thread', 'typing_extensions',
    'zoneinfo'
}

# Package name mappings (pypi name -> import name)
PACKAGE_IMPORT_MAP = {
    'fastapi': ['fastapi'],
    'uvicorn': ['uvicorn'],
    'pydantic': ['pydantic'],
    'requests': ['requests'],
    'httpx': ['httpx'],
    'opentelemetry-api': ['opentelemetry'],
    'opentelemetry-sdk': ['opentelemetry'],
    'opentelemetry-instrumentation': ['opentelemetry'],
    'opentelemetry-instrumentation-fastapi': ['opentelemetry'],
    'opentelemetry-instrumentation-requests': ['opentelemetry'],
    'opentelemetry-exporter-jaeger': ['opentelemetry'],
    'opentelemetry-exporter-prometheus': ['opentelemetry'],
    'prometheus_client': ['prometheus_client'],
    'prometheus-client': ['prometheus_client'],
    'pymongo': ['pymongo', 'bson'],
    'cryptography': ['cryptography'],
    'PyJWT': ['jwt'],
    'pyjwt': ['jwt'],
    'psutil': ['psutil'],
    'deprecated': ['deprecated'],
    'tabulate': ['tabulate'],
}

# Local project modules that may be imported but are not in requirements.txt
LOCAL_PROJECT_MODULES = {
    'config',       # Local configuration module
    'db',           # Local database module
    'ipsec',        # Local IPsec module
    'amf',          # AMF module
    'smf',          # SMF module
    'nrf',          # NRF module
    'upf',          # UPF module
    'udm',          # UDM module
    'udr',          # UDR module
    'ausf',         # AUSF module
    'pcf',          # PCF module
    'nssf',         # NSSF module
    'bsf',          # BSF module
    'chf',          # CHF module
    'scp',          # SCP module
    'sepp',         # SEPP module
    'nef',          # NEF module
    'n3iwf',        # N3IWF module
    'core_network', # Core network package
}


@dataclass
class Issue:
    """Represents a code quality issue."""
    file: str
    line: Optional[int]
    category: str
    severity: str  # 'error', 'warning', 'info'
    message: str


@dataclass
class CheckResult:
    """Result of a single check category."""
    name: str
    passed: bool
    issues: List[Issue] = field(default_factory=list)
    summary: str = ""


class CodeQualityChecker:
    """Static analysis checker for 5G Emulator API."""

    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.core_network_path = self.base_path / "core_network"
        self.requirements_path = self.base_path / "requirements.txt"
        self.issues: List[Issue] = []
        self.results: List[CheckResult] = []

        # NF files to analyze (exclude test files, backup files, utility scripts)
        self.nf_files = self._get_nf_files()

    def _get_nf_files(self) -> List[Path]:
        """Get list of NF Python files to analyze."""
        nf_files = []
        if self.core_network_path.exists():
            for py_file in self.core_network_path.glob("*.py"):
                # Exclude test files, backup files, and utility scripts
                filename = py_file.name
                if (not filename.startswith('test') and
                    not filename.endswith('.bak') and
                    not filename.startswith('amf-') and  # Exclude old metric scripts
                    filename not in ('test.py', 'db.py', 'ipsec.py', 'main.py')):
                    nf_files.append(py_file)
        return sorted(nf_files)

    def _parse_file(self, filepath: Path) -> Optional[ast.AST]:
        """Parse a Python file and return its AST."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            return ast.parse(content, filename=str(filepath))
        except SyntaxError as e:
            self.issues.append(Issue(
                file=str(filepath.relative_to(self.base_path)),
                line=e.lineno,
                category="syntax",
                severity="error",
                message=f"Syntax error: {e.msg}"
            ))
            return None
        except Exception as e:
            self.issues.append(Issue(
                file=str(filepath.relative_to(self.base_path)),
                line=None,
                category="parse",
                severity="error",
                message=f"Failed to parse file: {str(e)}"
            ))
            return None

    def _extract_imports(self, tree: ast.AST) -> Set[str]:
        """Extract all imports from an AST."""
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module.split('.')[0])
        return imports

    def _check_uvicorn_import(self, filepath: Path, tree: ast.AST) -> List[Issue]:
        """Check if file uses uvicorn.run() but doesn't import uvicorn."""
        issues = []

        # Read file content for text-based check
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        imports = self._extract_imports(tree)

        # Check for uvicorn.run() usage
        if 'uvicorn.run' in content and 'uvicorn' not in imports:
            issues.append(Issue(
                file=str(filepath.relative_to(self.base_path)),
                line=None,
                category="import",
                severity="error",
                message="Uses uvicorn.run() but 'import uvicorn' is missing"
            ))

        return issues

    def check_imports(self) -> CheckResult:
        """Check 1: Import Validation."""
        print("\n[1/5] Checking Import Validation...")
        issues = []

        common_imports = {
            'fastapi': ['FastAPI', 'HTTPException', 'Request', 'Depends'],
            'pydantic': ['BaseModel', 'Field'],
            'uvicorn': ['run'],
            'requests': ['post', 'get', 'RequestException'],
            'httpx': ['AsyncClient', 'Client'],
        }

        for nf_file in self.nf_files:
            tree = self._parse_file(nf_file)
            if tree is None:
                continue

            # Check uvicorn import
            issues.extend(self._check_uvicorn_import(nf_file, tree))

            # Read file content
            with open(nf_file, 'r', encoding='utf-8') as f:
                content = f.read()

            imports = self._extract_imports(tree)

            # Check for common missing imports
            for module, usages in common_imports.items():
                for usage in usages:
                    # Check if module.usage or just usage is used
                    if (f'{module}.{usage}' in content or
                        (usage in content and module not in imports)):
                        # More sophisticated check - look for actual usage
                        pattern = rf'\b{module}\.{usage}\b'
                        if re.search(pattern, content) and module not in imports:
                            issues.append(Issue(
                                file=str(nf_file.relative_to(self.base_path)),
                                line=None,
                                category="import",
                                severity="error",
                                message=f"Uses {module}.{usage}() but 'import {module}' is missing"
                            ))

        passed = len([i for i in issues if i.severity == 'error']) == 0
        return CheckResult(
            name="Import Validation",
            passed=passed,
            issues=issues,
            summary=f"Checked {len(self.nf_files)} files, found {len(issues)} import issues"
        )

    def _parse_requirements(self) -> Set[str]:
        """Parse requirements.txt and return set of available import names."""
        available_imports = set()

        if not self.requirements_path.exists():
            return available_imports

        with open(self.requirements_path, 'r') as f:
            for line in f:
                line = line.strip()
                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue

                # Extract package name (before version specifier)
                match = re.match(r'^([a-zA-Z0-9_-]+)', line)
                if match:
                    pkg_name = match.group(1).lower()

                    # Add mapped import names
                    for req_pkg, import_names in PACKAGE_IMPORT_MAP.items():
                        if req_pkg.lower() == pkg_name:
                            available_imports.update(import_names)
                            break
                    else:
                        # If no mapping, assume import name is same as package name
                        available_imports.add(pkg_name.replace('-', '_'))

        return available_imports

    def check_dependencies(self) -> CheckResult:
        """Check 2: Dependency Validation."""
        print("\n[2/5] Checking Dependency Validation...")
        issues = []

        available_imports = self._parse_requirements()

        for nf_file in self.nf_files:
            tree = self._parse_file(nf_file)
            if tree is None:
                continue

            imports = self._extract_imports(tree)

            # Check each import
            for imp in imports:
                # Skip stdlib modules
                if imp in STDLIB_MODULES:
                    continue

                # Skip local modules (relative imports)
                if imp.startswith('.'):
                    continue

                # Skip known local project modules
                if imp in LOCAL_PROJECT_MODULES:
                    continue

                # Check if import is available from requirements
                if imp.lower() not in {i.lower() for i in available_imports}:
                    # Check common mappings
                    found = False
                    for req_pkg, import_names in PACKAGE_IMPORT_MAP.items():
                        if imp in import_names:
                            found = True
                            break

                    if not found:
                        issues.append(Issue(
                            file=str(nf_file.relative_to(self.base_path)),
                            line=None,
                            category="dependency",
                            severity="warning",
                            message=f"Import '{imp}' not found in requirements.txt"
                        ))

        passed = len([i for i in issues if i.severity == 'error']) == 0
        return CheckResult(
            name="Dependency Validation",
            passed=passed,
            issues=issues,
            summary=f"Checked {len(self.nf_files)} files against {len(available_imports)} dependencies"
        )

    def check_health_endpoints(self) -> CheckResult:
        """Check 3: Health Endpoint Validation."""
        print("\n[3/5] Checking Health Endpoint Validation...")
        issues = []
        required_fields = {'status', 'service', 'compliance', 'version'}

        for nf_file in self.nf_files:
            with open(nf_file, 'r', encoding='utf-8') as f:
                content = f.read()

            tree = self._parse_file(nf_file)
            if tree is None:
                continue

            # Look for /health endpoint
            has_health_endpoint = False
            health_endpoint_line = None

            # Check for @app.get("/health") decorator pattern
            if re.search(r'@app\.(get|route)\s*\(\s*["\']/?health["\']', content):
                has_health_endpoint = True
                match = re.search(r'@app\.(get|route)\s*\(\s*["\']/?health["\']', content)
                if match:
                    health_endpoint_line = content[:match.start()].count('\n') + 1

            if not has_health_endpoint:
                issues.append(Issue(
                    file=str(nf_file.relative_to(self.base_path)),
                    line=None,
                    category="health_endpoint",
                    severity="error",
                    message="Missing /health endpoint"
                ))
                continue

            # Check for required fields in health response
            # Look for the function after the decorator
            health_pattern = r'@app\.(get|route)\s*\(\s*["\']/?health["\']\s*\).*?def\s+\w+.*?return\s*\{([^}]+)\}'
            match = re.search(health_pattern, content, re.DOTALL)

            if match:
                return_dict_content = match.group(2)
                missing_fields = []
                for field in required_fields:
                    if f'"{field}"' not in return_dict_content and f"'{field}'" not in return_dict_content:
                        missing_fields.append(field)

                if missing_fields:
                    issues.append(Issue(
                        file=str(nf_file.relative_to(self.base_path)),
                        line=health_endpoint_line,
                        category="health_endpoint",
                        severity="warning",
                        message=f"Health endpoint missing required fields: {', '.join(missing_fields)}"
                    ))

        passed = len([i for i in issues if i.severity == 'error']) == 0
        return CheckResult(
            name="Health Endpoint Validation",
            passed=passed,
            issues=issues,
            summary=f"Checked {len(self.nf_files)} files for health endpoints"
        )

    def check_port_configuration(self) -> CheckResult:
        """Check 4: Port Configuration Validation."""
        print("\n[4/5] Checking Port Configuration...")
        issues = []
        port_assignments: Dict[int, List[str]] = {}

        for nf_file in self.nf_files:
            with open(nf_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Look for default port in argparse
            # Pattern: parser.add_argument("--port", ..., default=XXXX, ...)
            port_pattern = r'add_argument\s*\([^)]*["\']--port["\'][^)]*default\s*=\s*(\d+)'
            match = re.search(port_pattern, content)

            if match:
                port = int(match.group(1))
                filename = nf_file.name

                if port not in port_assignments:
                    port_assignments[port] = []
                port_assignments[port].append(filename)

        # Check for port conflicts
        for port, files in port_assignments.items():
            if len(files) > 1:
                issues.append(Issue(
                    file="multiple",
                    line=None,
                    category="port_conflict",
                    severity="error",
                    message=f"Port {port} is used by multiple NFs: {', '.join(files)}"
                ))

        # Report port assignments for reference
        if port_assignments:
            print(f"   Port assignments found:")
            for port in sorted(port_assignments.keys()):
                files = port_assignments[port]
                status = "[CONFLICT]" if len(files) > 1 else "[OK]"
                print(f"     Port {port}: {', '.join(files)} {status}")

        passed = len([i for i in issues if i.severity == 'error']) == 0
        return CheckResult(
            name="Port Configuration Validation",
            passed=passed,
            issues=issues,
            summary=f"Found {len(port_assignments)} unique port assignments, {len(issues)} conflicts"
        )

    def check_main_block(self) -> CheckResult:
        """Check 5: Main Block Validation."""
        print("\n[5/5] Checking Main Block Validation...")
        issues = []

        for nf_file in self.nf_files:
            with open(nf_file, 'r', encoding='utf-8') as f:
                content = f.read()

            tree = self._parse_file(nf_file)
            if tree is None:
                continue

            # Check for if __name__ == "__main__": block
            has_main_block = False
            for node in ast.walk(tree):
                if isinstance(node, ast.If):
                    # Check if it's the __name__ == "__main__" pattern
                    if isinstance(node.test, ast.Compare):
                        if (isinstance(node.test.left, ast.Name) and
                            node.test.left.id == '__name__'):
                            for comparator in node.test.comparators:
                                if (isinstance(comparator, ast.Constant) and
                                    comparator.value == "__main__"):
                                    has_main_block = True
                                    break

            if not has_main_block:
                issues.append(Issue(
                    file=str(nf_file.relative_to(self.base_path)),
                    line=None,
                    category="main_block",
                    severity="error",
                    message="Missing 'if __name__ == \"__main__\":' block"
                ))
                continue

            # Check for argparse usage
            has_argparse = 'argparse' in self._extract_imports(tree)
            if not has_argparse:
                issues.append(Issue(
                    file=str(nf_file.relative_to(self.base_path)),
                    line=None,
                    category="main_block",
                    severity="warning",
                    message="Missing argparse import for CLI argument handling"
                ))
                continue

            # Check for --host and --port arguments
            has_host_arg = '--host' in content
            has_port_arg = '--port' in content

            if not has_host_arg:
                issues.append(Issue(
                    file=str(nf_file.relative_to(self.base_path)),
                    line=None,
                    category="main_block",
                    severity="warning",
                    message="Missing --host argument in argparse"
                ))

            if not has_port_arg:
                issues.append(Issue(
                    file=str(nf_file.relative_to(self.base_path)),
                    line=None,
                    category="main_block",
                    severity="warning",
                    message="Missing --port argument in argparse"
                ))

        passed = len([i for i in issues if i.severity == 'error']) == 0
        return CheckResult(
            name="Main Block Validation",
            passed=passed,
            issues=issues,
            summary=f"Checked {len(self.nf_files)} files for main block configuration"
        )

    def run_all_checks(self) -> bool:
        """Run all checks and return True if all pass."""
        print("=" * 70)
        print("5G Emulator API - Static Analysis Code Quality Check")
        print("=" * 70)
        print(f"Base path: {self.base_path}")
        print(f"Core network path: {self.core_network_path}")
        print(f"NF files to analyze: {len(self.nf_files)}")

        if not self.nf_files:
            print("\nERROR: No NF files found to analyze!")
            return False

        # Run all checks
        self.results.append(self.check_imports())
        self.results.append(self.check_dependencies())
        self.results.append(self.check_health_endpoints())
        self.results.append(self.check_port_configuration())
        self.results.append(self.check_main_block())

        # Print results
        print("\n" + "=" * 70)
        print("RESULTS SUMMARY")
        print("=" * 70)

        all_passed = True
        total_errors = 0
        total_warnings = 0

        for result in self.results:
            status = "[PASS]" if result.passed else "[FAIL]"
            errors = len([i for i in result.issues if i.severity == 'error'])
            warnings = len([i for i in result.issues if i.severity == 'warning'])

            print(f"\n{status} {result.name}")
            print(f"   {result.summary}")

            if not result.passed:
                all_passed = False

            total_errors += errors
            total_warnings += warnings

            # Print issues
            for issue in result.issues:
                severity_icon = "ERROR" if issue.severity == 'error' else "WARN "
                line_info = f":{issue.line}" if issue.line else ""
                print(f"   [{severity_icon}] {issue.file}{line_info}: {issue.message}")

        print("\n" + "=" * 70)
        print("FINAL RESULT")
        print("=" * 70)

        if all_passed:
            print(f"\n[PASS] All checks passed!")
        else:
            print(f"\n[FAIL] Some checks failed!")

        print(f"   Total errors: {total_errors}")
        print(f"   Total warnings: {total_warnings}")
        print(f"   Files analyzed: {len(self.nf_files)}")

        return all_passed


def main():
    """Main entry point."""
    # Determine base path
    script_path = Path(__file__).resolve()
    base_path = script_path.parent.parent  # Go up from scripts/ to project root

    # Check if we're in the right directory structure
    if not (base_path / "core_network").exists():
        print(f"ERROR: core_network directory not found at {base_path / 'core_network'}")
        print("Make sure to run this script from the project root or scripts directory")
        sys.exit(1)

    # Run checks
    checker = CodeQualityChecker(str(base_path))
    success = checker.run_all_checks()

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
