"""Conservative channel previews. Arbitrary shell arguments are never published."""
import os
import re
import shlex

HIDDEN = '[details hidden: sensitive data]'
_SENSITIVE = re.compile(
    r"(?i)(?:\.env\b|credential|secret|token|password|passwd|authorization|api[_-]?key|private[_-]?key|"
    r"\b(?:gh[pousr]_|sk-|eyJ)[A-Za-z0-9_.-]{12,}|cookie|BEGIN [A-Z ]*PRIVATE|"
    r"[a-z][a-z0-9+.-]*://[^\s/@]+@|[A-Za-z0-9_+=]{32,})"
)
_FLAGS = {'-q', '-v', '-vv', '-x', '-s', '-ra', '--run', '--noEmit', '--no-emit', '--stat', '--short',
          '--porcelain', '--name-only', '--name-status', '--check', '--cached', '--staged', '--oneline',
          '--all', '--max-count', '--', '-n', '-1', '-5', '-10', '--no-color', '--color', '--collect-only'}
_NAMES = {'src', 'tests', 'test', 'services', 'clients', 'libs', 'docs', 'dist', 'build', 'deploy',
          'HEAD', 'main', 'origin', 'package.json', 'package-lock.json', 'pyproject.toml'}


def _path(value):
    # Only workspace paths, never URLs, free text, environment expansion or shell programs.
    return bool(re.fullmatch(r'[A-Za-z0-9_./-]+', value)) and (
        value in _NAMES or value in {'.', '..'} or value.startswith(('./', '../', '/subjective/', '/tmp/'))
        or '/' in value and not value.startswith('/')
        or value.endswith(('.py', '.ts', '.tsx', '.js', '.json', '.toml', '.yml', '.yaml')))


def _preview_segment(words):
    if not words:
        return None
    executable = words[0].rsplit('/', 1)[-1]
    args = words[1:]
    allowed = False
    if executable == 'pwd':
        allowed = not args
    elif executable == 'cd':
        allowed = len(args) == 1 and _path(args[0])
    elif executable in {'ls', 'pytest', 'tsc', 'eslint'}:
        allowed = all(arg in _FLAGS or _path(arg) for arg in args)
    elif executable in {'python', 'python3'} and args[:2] in (['-m', 'pytest'], ['-m', 'compileall']):
        allowed = all(arg in _FLAGS or _path(arg) for arg in args[2:])
    elif executable == 'git' and args and args[0] in {'status', 'diff', 'log', 'show', 'rev-parse', 'ls-files'}:
        allowed = all(arg in _FLAGS or _path(arg) for arg in args[1:])
    elif executable == 'npm' and args:
        prefix = 2 if args[:1] == ['run'] else 1
        script = args[1] if prefix == 2 and len(args) > 1 else args[0]
        allowed = script in {'test', 'build', 'lint', 'typecheck', 'check'} and all(
            arg in _FLAGS or _path(arg) for arg in args[prefix:])
    elif executable == 'npx' and args and args[0] in {'vitest', 'tsc', 'eslint'}:
        allowed = all(arg in _FLAGS or _path(arg) or arg == 'run' for arg in args[1:])
    if allowed:
        return shlex.join(words)
    # Only display ordinary executable names; arbitrary executable paths may themselves contain data.
    name = executable if re.fullmatch(r'[A-Za-z][A-Za-z0-9_-]{0,39}', executable) else 'Shell command'
    return name + ' [arguments hidden]'


def safe_tool_detail(value, *, command=True):
    if not isinstance(value, str) or not value.strip():
        return None
    if value == HIDDEN:
        return HIDDEN
    if len(value) > 16000:
        return HIDDEN
    known_secret = any(len(secret) >= 6 and secret in value for key, secret in os.environ.items()
                       if re.search(r'(?i)(token|secret|password|api.?key|credential)', key))
    if _SENSITIVE.search(value) or known_secret:
        return HIDDEN
    if not command:
        return re.sub(r'[\x00-\x1f\x7f]', ' ', value).strip()[:500]
    try:
        lexer = shlex.shlex(value, posix=True, punctuation_chars=';&|<>()')
        lexer.whitespace_split = True
        lexer.commenters = ''
        words = list(lexer)
    except ValueError:
        return HIDDEN
    if _SENSITIVE.search(' '.join(words)):
        return HIDDEN
    # No substitutions, redirections, heredocs or nested programs in visible previews.
    if any(char in value for char in ('$', '`', '\n', '\r')) or any(word in {'<', '>', '>>', '<<', '<<<', '(', ')'} for word in words):
        return 'Shell command [arguments hidden]'
    parts, segment = [], []
    for word in words:
        if word in {'&&', '||', ';', '|'}:
            preview = _preview_segment(segment)
            if not preview:
                return 'Shell command [arguments hidden]'
            parts.extend([preview, word]); segment = []
        else:
            segment.append(word)
    preview = _preview_segment(segment)
    if not preview:
        return 'Shell command [arguments hidden]'
    parts.append(preview)
    result = ' '.join(parts)
    return result if len(result) <= 500 else result[:499] + '…'
