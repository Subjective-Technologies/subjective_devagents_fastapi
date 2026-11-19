"""Runtime capability profiles: what an execution can run, chosen from the assigned action.

A logical role (Integration, Coder) is not an execution capability. A verifier assigned
"run tests, run tsc, run build" needs test/build execution and the Node version the
project pins, not the host default. This module

* discovers installed toolchains (Node from nvm and system paths, Python, Go),
* resolves a project's Node pin from .nvmrc, .node-version, package.json engines and
  CI workflow node-version (in that order),
* uses approved stage requirements, with a fallback for older assignments,
* selects a compatible profile, or reports BLOCKED_NO_COMPATIBLE_RUNTIME at once,
* classifies verification commands (tests, type checks, builds) that a read-only
  verifier may run without source-editing tools.

Nothing here grants write access to sources. A verify profile keeps Write/Edit absent;
build commands may still produce ignored build artifacts.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import threading
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from .language_models.contracts import ModelError

_VERSION = re.compile(r'^v?(\d+)(?:\.(\d+))?(?:\.(\d+))?')
_CACHE_LOCK = threading.Lock()
_TOOLCHAINS: dict[str, list] = {}
_PINS: dict[tuple, object] = {}


@dataclass(frozen=True)
class NodeToolchain:
    version: tuple[int, int, int]
    bin_dir: str
    source: str

    @property
    def major(self) -> int:
        return self.version[0]

    def label(self) -> str:
        return 'v' + '.'.join(str(part) for part in self.version)


@dataclass
class Requirements:
    stage: str | None = None
    capabilities: frozenset[str] = field(default_factory=lambda: frozenset({'fs_read'}))
    test_exec: bool = False
    build_exec: bool = False
    browser_exec: bool = False
    write: bool = False
    node_majors: set[int] = field(default_factory=set)
    node_sources: dict[int, str] = field(default_factory=dict)

    def summary(self) -> dict:
        return {'stage': self.stage, 'required_runtime_capabilities': sorted(self.runtime_capabilities()),
                'test_exec': self.test_exec, 'build_exec': self.build_exec, 'browser_exec': self.browser_exec,
                'write': self.write, 'node_majors': sorted(self.node_majors),
                'node_sources': {str(k): v for k, v in self.node_sources.items()}}

    def runtime_capabilities(self) -> frozenset[str]:
        return self.capabilities | {name for name, needed in (
            ('fs_write', self.write), ('test_exec', self.test_exec), ('build_exec', self.build_exec),
            ('browser_exec', self.browser_exec),
        ) if needed}


@dataclass(frozen=True)
class RuntimeProfile:
    runtime_id: str
    sandbox: str                     # 'read-only' | 'workspace-write'
    node_versions: tuple[str, ...]
    shell_exec: bool = True
    fs_read: bool = True
    fs_write: bool = False
    git_read: bool = True
    git_write: bool = False
    test_exec: bool = False
    build_exec: bool = False
    browser_exec: bool = False
    network_access: bool = True
    python_versions: tuple[str, ...] = ()
    go_available: bool = False

    def as_dict(self) -> dict:
        return asdict(self)


CAPABILITY_ALIASES = {'shell': 'shell_exec', 'filesystem': 'fs_read'}
RUNTIME_CAPABILITIES = frozenset({
    'shell_exec', 'fs_read', 'fs_write', 'git_read', 'git_write', 'test_exec',
    'build_exec', 'browser_exec', 'network_access',
})


def _parse_version(text):
    match = _VERSION.match(str(text or '').strip())
    if not match:
        return None
    return tuple(int(part or 0) for part in match.groups())


def _nvm_roots():
    roots = []
    configured = os.environ.get('SUBJECTIVE_RUNTIME_NODE_DIRS', '')
    for item in configured.split(os.pathsep):
        if item:
            roots.append(Path(item).expanduser())
    for base in filter(None, [os.environ.get('NVM_DIR'), str(Path.home() / '.nvm')]):
        roots.append(Path(base) / 'versions' / 'node')
    return roots


def node_toolchains(refresh=False) -> list[NodeToolchain]:
    """Installed Node toolchains, newest first. Directory names give nvm versions."""
    with _CACHE_LOCK:
        if not refresh and 'node' in _TOOLCHAINS:
            return list(_TOOLCHAINS['node'])
    found: dict[str, NodeToolchain] = {}
    for root in _nvm_roots():
        try:
            entries = sorted(root.iterdir()) if root.is_dir() else []
        except OSError:
            entries = []
        for entry in entries:
            version = _parse_version(entry.name)
            binary = entry / 'bin' / 'node'
            if version and os.access(binary, os.X_OK):
                found.setdefault(str(entry / 'bin'), NodeToolchain(version, str(entry / 'bin'), 'nvm'))
    seen = {os.path.realpath(Path(path) / 'node') for path in found}
    for directory in ('/usr/local/bin', '/usr/bin', '/bin'):
        binary = Path(directory) / 'node'
        # /bin is often a symlink to /usr/bin: one toolchain, listed once.
        if not os.access(binary, os.X_OK) or os.path.realpath(binary) in seen:
            continue
        seen.add(os.path.realpath(binary))
        try:
            output = subprocess.run([str(binary), '--version'], capture_output=True, text=True, timeout=5,
                                    env={'PATH': '/usr/bin:/bin'}).stdout
        except (OSError, subprocess.SubprocessError):
            continue
        version = _parse_version(output)
        if version:
            found[directory] = NodeToolchain(version, directory, 'system')
    result = sorted(found.values(), key=lambda item: item.version, reverse=True)
    with _CACHE_LOCK:
        _TOOLCHAINS['node'] = result
    return list(result)


def python_versions() -> tuple[str, ...]:
    names = []
    for name in ('python3', 'python3.13', 'python3.12', 'python3.11', 'python3.10'):
        if shutil.which(name, path='/usr/local/bin:/usr/bin:/bin'):
            names.append(name)
    return tuple(names)


# ---- project pins -------------------------------------------------------------------

def _major_constraints(spec):
    """Parse a semver range into alternatives of (lower, upper_exclusive) major bounds."""
    spec = str(spec or '').strip()
    if not spec or spec in {'*', 'x', 'latest', 'node', 'current'}:
        return None
    if spec.startswith('lts'):
        return None
    alternatives = []
    for option in spec.split('||'):
        low, high = 0, 10_000
        tokens = re.findall(r'(>=|<=|>|<|\^|~|=)?\s*v?(\d+)(?:\.(\d+|x|\*))?(?:\.(\d+|x|\*))?', option)
        if not tokens:
            continue
        hyphen = re.match(r'\s*v?(\d+)\S*\s+-\s+v?(\d+)', option)
        if hyphen:
            alternatives.append((int(hyphen.group(1)), int(hyphen.group(2)) + 1))
            continue
        for operator, major, minor, patch in tokens:
            major = int(major)
            if operator in {'', '=', '^', '~'}:
                low, high = max(low, major), min(high, major + 1)
            elif operator == '>=':
                low = max(low, major)
            elif operator == '>':
                low = max(low, major + (0 if (minor and minor not in {'x', '*'}) or patch else 1))
            elif operator == '<':
                high = min(high, major + (1 if (minor and minor not in {'0', 'x', '*'}) or (patch and patch != '0') else 0))
            elif operator == '<=':
                high = min(high, major + 1)
        if low < high:
            alternatives.append((low, high))
    return alternatives or None


def _satisfies(major, alternatives):
    return alternatives is None or any(low <= major < high for low, high in alternatives)


def _read(path, limit=200_000):
    try:
        with open(path, 'r', encoding='utf-8', errors='replace') as stream:
            return stream.read(limit)
    except OSError:
        return None


def project_node_pin(directory, root=None, *, include_ci=True):
    """(alternatives, source) for the nearest pin between directory and root, or (None, None).

    .nvmrc / .node-version win, then package.json engines.node, then the repository's
    CI workflow node-version. The search never leaves the nearest Git repository (a
    submodule is its own project; the monorepo's security-scan workflow is not its
    toolchain). Results are cached per directory and file mtimes.
    """
    start = Path(directory).resolve()
    stop = Path(root).resolve() if root else None
    current = start
    for _ in range(12):
        key = (str(current), include_ci)
        stamp = []
        for name in ('.nvmrc', '.node-version', 'package.json'):
            try:
                stamp.append((current / name).stat().st_mtime)
            except OSError:
                stamp.append(None)
        cached = _PINS.get(key)
        if cached and cached[0] == stamp and cached[1][0] is not None:
            return cached[1]
        result = (None, None)
        for name in ('.nvmrc', '.node-version'):
            content = _read(current / name, 200)
            if content and content.strip():
                constraints = _major_constraints(content.strip().split()[0])
                if constraints:
                    result = (constraints, str(current / name))
                    break
        if result[0] is None:
            content = _read(current / 'package.json')
            if content:
                try:
                    engines = (json.loads(content).get('engines') or {}).get('node')
                except (ValueError, AttributeError):
                    engines = None
                constraints = _major_constraints(engines) if isinstance(engines, str) else None
                if constraints:
                    result = (constraints, str(current / 'package.json') + '#engines.node')
        repo_root = (current / '.git').exists()
        if result[0] is None and include_ci and repo_root:
            workflows = current / '.github' / 'workflows'
            try:
                files = sorted(workflows.glob('*.y*ml')) if workflows.is_dir() else []
            except OSError:
                files = []
            for item in files:
                match = re.search(r'node-version:\s*[\'"]?v?(\d+)', _read(item) or '')
                if match:
                    major = int(match.group(1))
                    result = ([(major, major + 1)], str(item) + '#node-version')
                    break
        _PINS[key] = (stamp, result)
        if result[0] is not None:
            return result
        if repo_root or current == stop or current.parent == current or (stop and not current.is_relative_to(stop)):
            break
        current = current.parent
    return None, None


def select_node(alternatives, toolchains=None):
    toolchains = node_toolchains() if toolchains is None else toolchains
    for toolchain in toolchains:   # newest first
        if _satisfies(toolchain.major, alternatives):
            return toolchain
    return None


def node_path_for(workdir, workspace, base_path):
    """PATH for a command in workdir: the project's pinned Node first, else unchanged."""
    alternatives, _ = project_node_pin(workdir, workspace)
    if alternatives is None:
        return base_path, None
    toolchain = select_node(alternatives)
    if toolchain is None:
        return base_path, None
    parts = [part for part in base_path.split(':') if part and part != toolchain.bin_dir]
    return ':'.join([toolchain.bin_dir, *parts]), toolchain


# ---- requirements ---------------------------------------------------------------------

_TEST = re.compile(r'(?i)\b(?:run(?:ning)?\s+(?:the\s+)?(?:\w+\s+){0,3}tests?|pytest|vitest|jest|'
                   r'npm\s+(?:run\s+)?test|go\s+test|cargo\s+test|npx\s+playwright\s+test)\b')
_BROWSER = re.compile(r'(?i)\b(?:(?:run|execute|exercise|verify|test|open|launch)\s+'
                      r'[^.!?\n]{0,55}\b(?:browsers?|playwright|selenium|cypress|puppeteer)|'
                      r'click\s+(?:the\s+)?(?:button|link|ui)|npx\s+playwright\s+test)\b')
_WRITE = re.compile(r'(?i)^(?:please\s+)?(?:implement|fix|patch|edit|refactor|add|update|remove|create|write)\b')
_BUILD = re.compile(r'(?i)\b(?:run(?:ning)?\s+(?:the\s+)?(?:\w+\s+){0,2}build|npm\s+run\s+build|tsc\b|typecheck|'
                    r'type[- ]check|vite\s+build|go\s+build|cargo\s+build|compile)')
_BROWSER_TOOL = re.compile(r'(?i)(?:^|[ /])(?:playwright|selenium|cypress|puppeteer|chromedriver|'
                           r'chromium(?:-browser)?|google-chrome)(?=[\s;|&]|$)')
# Only an instruction names a requirement ("run Node 22 tests", "on Node 22", "requires
# Node 24"); a mention in quoted history ("Integration got Node 18") is not one.
_NODE = re.compile(r'(?i)\b(?:on|with|under|using|use|requires?|run|pinned\s+to|needs?)\s+'
                   r'(?:node(?:\.js)?|nodejs)\s*(?:version\s*)?v?(\d{2})(?:\.\d+)*\b')
_PATHLIKE = re.compile(r'(?<![\w@:/.-])((?:/subjective/)?(?:clients|services|libs|apps|tools|websites|worktwins)/[\w.-]+(?:/[\w.-]+)?)')


def requirements_for(assignment, workspace=None, *, verifier=False, stage=None,
                     capabilities=None) -> Requirements:
    """Use the approved stage contract, or infer direct actions from an old assignment."""
    if capabilities is not None:
        if not isinstance(capabilities, list) or any(not isinstance(name, str) for name in capabilities):
            raise ModelError('BLOCKED_NO_COMPATIBLE_RUNTIME', 'Invalid approved runtime capabilities')
        required = frozenset(CAPABILITY_ALIASES.get(name, name) for name in capabilities)
        if not required <= RUNTIME_CAPABILITIES:
            raise ModelError('BLOCKED_NO_COMPATIBLE_RUNTIME', 'Unknown approved runtime capability')
        return Requirements(stage=stage, capabilities=required, write='fs_write' in required,
                            test_exec='test_exec' in required, build_exec='build_exec' in required,
                            browser_exec='browser_exec' in required)
    text = str(assignment or '')[-6000:]
    directives = '\n'.join(part.strip() for part in re.split(r'[.!?](?=\s|$)|\n', text)
                           if not re.search(r'(?i)\b(?:later|eventually|will|before DONE)\b', part))
    result = Requirements(test_exec=bool(_TEST.search(directives)) or verifier,
                          build_exec=bool(_BUILD.search(directives)) or verifier,
                          browser_exec=bool(_BROWSER.search(directives)), write=bool(_WRITE.match(directives)))
    for match in _NODE.finditer(directives):
        major = int(match.group(1))
        if 10 <= major <= 60:
            result.node_majors.add(major)
            result.node_sources.setdefault(major, 'assignment')
    if workspace:
        root = Path(workspace).resolve()
        seen = set()
        for match in _PATHLIKE.finditer(text):
            relative = match.group(1).removeprefix('/subjective/')
            candidate = (root / relative).resolve()
            if candidate in seen or not candidate.is_relative_to(root) or not candidate.is_dir():
                continue
            seen.add(candidate)
            if len(seen) > 20:
                break
            # Blocking needs an explicit project pin; CI hints only steer PATH selection.
            alternatives, source = project_node_pin(candidate, root, include_ci=False)
            if alternatives:
                for low, high in alternatives[:1]:
                    if high - low == 1:
                        result.node_majors.add(low)
                        result.node_sources.setdefault(low, source)
    return result


def browser_runtime_validated() -> bool:
    """True only when the operator marked this host and a browser plus a driver exist.

    A package name in an agent prompt is not validation. Unset SUBJECTIVE_BROWSER_EXEC
    means no profile advertises browser_exec.
    """
    if os.environ.get('SUBJECTIVE_BROWSER_EXEC') != 'validated':
        return False
    browser = shutil.which('chromium') or shutil.which('google-chrome') or shutil.which('chromium-browser')
    driver = shutil.which('chromedriver') or shutil.which('playwright')
    return bool(browser and driver)


def available_profiles(identity_id, sandbox, *, verify_allowed) -> list[RuntimeProfile]:
    nodes = tuple(item.label() for item in node_toolchains())
    pythons = python_versions()
    go = bool(shutil.which('go', path='/usr/local/go/bin:/usr/local/bin:/usr/bin:/bin'))
    browser = browser_runtime_validated()
    profiles = []
    if sandbox == 'workspace-write':
        profiles.append(RuntimeProfile(identity_id, 'workspace-write', nodes, fs_write=True, git_write=True,
                                       test_exec=True, build_exec=True,
                                       python_versions=pythons, go_available=go))
    if verify_allowed or sandbox == 'workspace-write':
        profiles.append(RuntimeProfile(identity_id + ':verify', 'read-only', nodes, test_exec=True, build_exec=True,
                                       python_versions=pythons, go_available=go))
    if browser:
        profiles.extend(replace(profile, runtime_id=profile.runtime_id + ':browser', browser_exec=True)
                        for profile in list(profiles))
    profiles.append(RuntimeProfile(identity_id + ':read', 'read-only', nodes, test_exec=False, build_exec=False,
                                   browser_exec=False, python_versions=pythons, go_available=go))
    return profiles


def select_profile(requirements: Requirements, profiles: list[RuntimeProfile], *, grants=None) -> RuntimeProfile:
    """The least-privileged compatible profile, or BLOCKED_NO_COMPATIBLE_RUNTIME."""
    reasons = []
    required = requirements.runtime_capabilities()
    if grants is not None:
        permitted = {CAPABILITY_ALIASES.get(name, name) for name in grants}
        missing = required - permitted
        if missing:
            raise ModelError('BLOCKED_NO_COMPATIBLE_RUNTIME', 'Agent is not authorized for ' + ', '.join(sorted(missing)))
        profiles = [RuntimeProfile(**{**profile.as_dict(), **{
            name: False for name in RUNTIME_CAPABILITIES - permitted
        }}) for profile in profiles]
    ordered = sorted(profiles, key=lambda p: (p.sandbox == 'workspace-write', p.browser_exec, p.build_exec, p.test_exec))
    for profile in ordered:
        installed = {(_parse_version(v) or (0,))[0] for v in profile.node_versions}
        missing = sorted(requirements.node_majors - installed)
        unavailable = [name for name in sorted(required) if not getattr(profile, name)]
        if unavailable:
            reasons.append(f'{profile.runtime_id}: ' + ', '.join(
                {'fs_write': 'no write access', 'test_exec': 'no test execution', 'build_exec': 'no build execution'}
                .get(name, 'missing ' + name) for name in unavailable))
            continue
        if missing:
            reasons.append(f'{profile.runtime_id}: Node {"/".join(map(str, missing))} not installed')
            continue
        return profile
    detail = '; '.join(reasons) or 'no runtime profile'
    raise ModelError('BLOCKED_NO_COMPATIBLE_RUNTIME', 'No compatible runtime: ' + detail[:900])


# ---- verification commands -----------------------------------------------------------

_SCRIPTS = {'test', 'tests', 'build', 'lint', 'typecheck', 'type-check', 'types', 'check', 'tsc', 'vitest',
            'test:unit', 'test:ci', 'test:integration', 'test:e2e', 'e2e', 'verify', 'ci', 'format:check'}
_NPX_TOOLS = {'tsc', 'vitest', 'jest', 'eslint', 'playwright', 'vite', 'prettier', 'tsx'}
_FILTERS = {'tail', 'head', 'grep', 'wc', 'sort', 'uniq', 'cat'}


def _verification_segment(tokens):
    while tokens and re.match(r'^[A-Za-z_][A-Za-z0-9_]*=', tokens[0]):
        tokens = tokens[1:]
    if not tokens:
        return 'prefix'
    base = tokens[0].rsplit('/', 1)[-1]
    rest = [token for token in tokens[1:] if not token.startswith('-')]
    if base == 'cd' or (base in {'source', '.'} and tokens[1:2] and tokens[1].endswith('nvm.sh')):
        return 'prefix'
    if base == 'nvm':
        if rest[:1] == ['use'] or rest[:1] == ['current'] or tokens[1:2] == ['--version']:
            return 'prefix'
        if rest[:1] in (['exec'], ['run']) and len(tokens) > 3:
            return _verification_segment(tokens[3:]) if rest[:1] == ['exec'] else 'command'
        return None
    if base in {'node'} and set(tokens[1:]) <= {'--version', '-v'}:
        return 'prefix'
    if base in {'npm', 'pnpm', 'yarn'}:
        args = [token for token in tokens[1:] if not token.startswith('-')]
        if args[:1] in (['test'], ['t']):
            return 'command'
        if args[:1] == ['run'] and len(args) > 1 and args[1] in _SCRIPTS:
            return 'command'
        if base in {'pnpm', 'yarn'} and args[:1] and args[0] in _SCRIPTS:
            return 'command'
        if args[:1] == ['exec'] and len(args) > 1 and args[1] in _NPX_TOOLS:
            return 'command'
        return None
    if base == 'npx':
        args = [token for token in tokens[1:] if not token.startswith('-')]
        return 'command' if args[:1] and args[0] in _NPX_TOOLS else None
    if base in {'tsc', 'vitest', 'jest', 'eslint'}:
        return 'command'
    if base in {'pytest', 'py.test'}:
        return 'command'
    if re.fullmatch(r'python(?:[0-9]+(?:\.[0-9]+)*)?', base) and tokens[1:3] in (['-m', 'pytest'], ['-m', 'unittest'], ['-m', 'compileall']):
        return 'command'
    if base == 'go' and rest[:1] and rest[0] in {'test', 'build', 'vet'}:
        return 'command'
    if base == 'cargo' and rest[:1] and rest[0] in {'test', 'build', 'check', 'clippy'}:
        return 'command'
    if base == 'make' and rest and all(target in {'test', 'build', 'check', 'lint', 'verify'} for target in rest):
        return 'command'
    return None


def is_verification_command(command) -> bool:
    """A test/type-check/build command a verify profile may run without a write lease.

    Allows `cd dir && …`, sourcing nvm and `nvm use N`, `&&` chaining of verification
    commands, `2>&1`, and pipes into simple filters. Rejects other redirection,
    substitution, background jobs, `;` and `||` sequencing.
    """
    command = str(command or '').strip()
    if not command or len(command) > 2000 or '\n' in command:
        return False
    cleaned = command.replace('2>&1', '')
    if re.search(r'[`$<>();]|\|\||(?<!&)&(?!&)', cleaned):
        return False
    pipeline = [part.strip() for part in command.split('|')]
    head, filters = pipeline[0], pipeline[1:]
    for item in filters:
        try:
            tokens = shlex.split(item.replace('2>&1', ''))
        except ValueError:
            return False
        if not tokens or tokens[0] not in _FILTERS or any(t in {'-i', '--in-place'} for t in tokens):
            return False
    saw_command = False
    for part in head.split('&&'):
        try:
            tokens = shlex.split(part.replace('2>&1', '').strip())
        except ValueError:
            return False
        kind = _verification_segment(tokens)
        if kind is None:
            return False
        saw_command = saw_command or kind == 'command'
    return saw_command


def verification_capability(command):
    if _BROWSER_TOOL.search(command):
        return 'browser_exec'
    if is_verification_command(command):
        return 'build_exec' if _BUILD.search(command) else 'test_exec'
    return None
