# Language Model Catalog Configuration

## Overview

This directory contains the managed model catalog for the DevAgents FastAPI service. The catalog defines which language model identities are available for use in the system.

**Architecture Decision (TAGGY-3451):** Due to runtime constraints (CLI binaries not consistently available for model discovery), we use a **managed model catalog** approach where model identities are defined in a version-controlled configuration file rather than discovered dynamically at runtime.

## Configuration File

### `language-models.json`

This is the canonical source of truth for available language model identities. Each identity represents one specific provider+model combination that users can select from the UI.

**Schema:**
```json
{
  "$schema": "language-models-config-v1",
  "$comment": "Optional description",
  "updated_at": "YYYY-MM-DD",
  "identities": [
    {
      "id": "unique-identifier",
      "provider": "claude|codex|cursor|grok",
      "model": "model-name",
      "strategy": "cli|api|api_agent",
      "home": "/absolute/path/to/home",
      "config_dir": "/absolute/path/to/config",
      "workspace": "/absolute/path/to/workspace",
      "executable": "/absolute/path/to/binary",
      "os_user": "username",
      "sandbox": "read-only|workspace-write",
      "allowed_tools": ["Read", "Write", "Edit", "Bash", "Glob", "Grep"]
    }
  ]
}
```

**Field Descriptions:**

- `id` (required): Unique identifier for this identity. Should be descriptive (e.g., `claude-opus-4`, `claude-sonnet-4-5`)
- `provider` (required): Provider name - must be one of: `claude`, `codex`, `cursor`, `grok`
- `model` (required for API strategies): Specific model identifier (e.g., `claude-opus-4`, `claude-sonnet-4-5`)
- `strategy` (default: `cli`): Execution strategy - `cli`, `api`, or `api_agent`
- `home`: Absolute path to the provider's home directory
- `config_dir`: Absolute path to the provider's configuration directory
- `workspace`: Absolute path to the workspace directory for tool execution
- `executable`: Absolute path to the CLI binary
- `os_user`: OS user to run as (for isolation)
- `sandbox`: Sandbox mode - `read-only` or `workspace-write`
- `allowed_tools`: Array of tool names this identity can use

**Provider-Specific Requirements:**

- **Claude**: `config_dir` must be `$HOME/.claude`
- **Cursor**: `config_dir` must be `$HOME/.config/cursor`
- **Grok**: `config_dir` must be `$HOME/.grok`

## API Endpoint

The catalog is served via the FastAPI backend at:

```
GET /api/language-model/identities
```

**Response Format:**
```json
{
  "identities": [
    {
      "id": "claude-opus-4",
      "provider": "claude",
      "model": "claude-opus-4",
      "strategy": "cli",
      "remote": false,
      "capabilities": {
        "text": true,
        "attachments": true,
        "interactive_approvals": false,
        "native_resume": true
      }
    }
  ]
}
```

## Deployment

### 1. Configuration File Location

The configuration file should be at:
```
/subjective/services/subjective_devagents_fastapi/config/language-models.json
```

### 2. Environment Variable

Set the `SUBJECTIVE_LANGUAGE_MODELS_CONFIG` environment variable to point to the configuration file:

```bash
export SUBJECTIVE_LANGUAGE_MODELS_CONFIG=/subjective/services/subjective_devagents_fastapi/config/language-models.json
```

**For systemd service:**
Add to `/etc/systemd/system/subjective-devagents-fastapi.service`:
```ini
[Service]
Environment="SUBJECTIVE_LANGUAGE_MODELS_CONFIG=/subjective/services/subjective_devagents_fastapi/config/language-models.json"
```

**For run.sh:**
Add to `/subjective/services/subjective_devagents_fastapi/run.sh`:
```bash
export SUBJECTIVE_LANGUAGE_MODELS_CONFIG="${SUBJECTIVE_LANGUAGE_MODELS_CONFIG:-$(dirname "$0")/config/language-models.json}"
```

### 3. Service Restart

After updating the configuration or environment variable:
```bash
sudo systemctl daemon-reload
sudo systemctl restart subjective-devagents-fastapi
```

### 4. Verification

Verify the catalog is loaded:
```bash
curl -sk https://127.0.0.1:8080/api/language-model/identities | jq
```

Expected output should show all configured identities.

## Desktop Client Integration

The desktop client (`subjective_chat`) fetches identities from the FastAPI backend:

**Environment Variable** (optional):
```bash
export DEVAGENTS_API_URL=https://127.0.0.1:8080
```

The client will automatically:
1. Fetch identities from `${DEVAGENTS_API_URL}/api/language-model/identities`
2. Populate the LLM provider/model dropdowns
3. Handle errors gracefully (empty list on failure)

## Updating Models

To add/remove/update available models:

1. **Edit the configuration file:**
   ```bash
   vi /subjective/services/subjective_devagents_fastapi/config/language-models.json
   ```

2. **Validate JSON syntax:**
   ```bash
   jq . /subjective/services/subjective_devagents_fastapi/config/language-models.json
   ```

3. **Commit changes** (version control):
   ```bash
   cd /subjective/services/subjective_devagents_fastapi
   git add config/language-models.json
   git commit -m "Update language model catalog: [description]"
   ```

4. **Restart the service:**
   ```bash
   sudo systemctl restart subjective-devagents-fastapi
   ```

5. **Verify the changes:**
   ```bash
   curl -sk https://127.0.0.1:8080/api/language-model/identities | jq '.identities[] | {id, provider, model}'
   ```

## Current Models (as of 2026-10-06)

- **Claude**: Opus 4, Sonnet 4.5, Haiku 3.5
- **Codex**: Standard
- **Cursor**: Default
- **Grok**: 2, 2 Mini

## Security Considerations

1. **Path Validation**: All paths (home, config_dir, workspace, executable) must be absolute and cannot contain parent traversal (`..`)
2. **User Isolation**: Each identity should ideally run as a separate OS user for security isolation
3. **Sandbox Mode**: Use `read-only` for untrusted workspaces, `workspace-write` for trusted environments
4. **Allowed Tools**: Restrict tools to minimum necessary set (e.g., no `Bash` for read-only scenarios)

## Troubleshooting

### Catalog Not Loading

**Symptom**: Desktop client shows no models available

**Check:**
1. Environment variable is set correctly
2. Configuration file exists and is readable
3. JSON syntax is valid
4. FastAPI service is running
5. Check logs: `/var/log/subjective-devagents-fastapi/`

**Test API directly:**
```bash
curl -sk https://127.0.0.1:8080/api/language-model/identities
```

### Invalid Identity Configuration

**Symptom**: Service fails to start or returns 503

**Check:**
1. All required fields are present
2. Provider names are valid (`claude`, `codex`, `cursor`, `grok`)
3. Paths are absolute and exist
4. OS users exist
5. Executables are present and executable

### CLI Binary Not Found

**Symptom**: Model execution fails with "executable not found"

**Solution:**
1. Verify the `executable` path in the config
2. Check binary exists: `ls -la /path/to/binary`
3. Check permissions: Binary must be executable
4. For cursor: typically at `/home/subjective/.local/bin/cursor`
5. For others: typically at `/usr/local/bin/{provider}`

## Architecture Notes

This configuration-driven approach was chosen for TAGGY-3451 due to:

1. **Runtime Constraints**: CLI binaries not consistently available for dynamic discovery
2. **Reliability**: Predictable, version-controlled model availability
3. **Flexibility**: Easy to update without code changes
4. **Security**: Operator controls exactly which models are available
5. **Simplicity**: Single source of truth, no complex discovery logic

For future enhancement to true dynamic discovery, the architecture supports:
- Adding a CLI model discovery layer
- Implementing provider-specific adapters
- Falling back to this catalog when discovery fails

## Related Documentation

- `/subjective/services/subjective_devagents_fastapi/LANGUAGE_MODEL.md` - Language Model API specification
- `/subjective/services/subjective_devagents_fastapi/examples/language-models.json` - Example configuration
- `src/subjective_devagents_fastapi/language_models/` - Implementation source code
