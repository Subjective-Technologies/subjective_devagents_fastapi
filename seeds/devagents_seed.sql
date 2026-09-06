-- Catalog seed for the `devagents` D1 database.
--
-- Providers and models only — no conversation data. Safe to re-run: every statement is an
-- upsert keyed on the primary key, so re-seeding refreshes prices and capabilities without
-- duplicating rows or touching anything a conversation points at.
--
-- Prices are USD per MILLION tokens, list price at the seed date. They are inputs to the
-- cost calculation, not history: usage_records snapshot the cost they computed, so editing
-- a price here never rewrites past spend.

-- The three CLIs we drive under a subscription seat, plus the metered APIs behind them.
INSERT INTO providers (
  id, display_name, vendor, kind, access_modes,
  supports_resume, supports_thinking, supports_tools, supports_attachments,
  icon_path, status, created_at, updated_at
) VALUES
  ('codex',     'Codex',   'openai',    'cli', 'subscription,api', 1, 1, 1, 1, '../assets/providers/codex.svg',   'active', '2026-09-06T00:00:00Z', '2026-09-06T00:00:00Z'),
  ('claude',    'Claude',  'anthropic', 'cli', 'subscription,api', 1, 1, 1, 1, '../assets/providers/claude.svg',  'active', '2026-09-06T00:00:00Z', '2026-09-06T00:00:00Z'),
  ('cursor',    'Cursor',  'cursor',    'ide', 'subscription',     1, 0, 1, 0, '../assets/providers/cursor.svg',  'active', '2026-09-06T00:00:00Z', '2026-09-06T00:00:00Z'),
  ('anthropic', 'Claude API', 'anthropic', 'api', 'api',           1, 1, 1, 1, '../assets/providers/claude.svg',  'active', '2026-09-06T00:00:00Z', '2026-09-06T00:00:00Z'),
  ('openai',    'OpenAI API', 'openai',    'api', 'api',           1, 1, 1, 1, '../assets/providers/chatgpt.svg', 'active', '2026-09-06T00:00:00Z', '2026-09-06T00:00:00Z'),
  ('gemini',    'Gemini',  'google',    'api', 'api',              1, 1, 1, 1, '../assets/providers/gemini.svg',  'active', '2026-09-06T00:00:00Z', '2026-09-06T00:00:00Z'),
  ('grok',      'Grok',    'xai',       'api', 'api',              1, 0, 1, 0, '../assets/providers/grok.svg',    'active', '2026-09-06T00:00:00Z', '2026-09-06T00:00:00Z')
ON CONFLICT (id) DO UPDATE SET
  display_name = excluded.display_name,
  vendor = excluded.vendor,
  kind = excluded.kind,
  access_modes = excluded.access_modes,
  supports_resume = excluded.supports_resume,
  supports_thinking = excluded.supports_thinking,
  supports_tools = excluded.supports_tools,
  supports_attachments = excluded.supports_attachments,
  icon_path = excluded.icon_path,
  status = excluded.status,
  updated_at = excluded.updated_at;

INSERT INTO models (
  id, provider_id, display_name, family,
  context_window_tokens, max_output_tokens,
  price_input_per_mtok, price_output_per_mtok, price_cache_read_per_mtok, price_cache_write_per_mtok,
  supports_reasoning, supports_vision, is_default, status, created_at, updated_at
) VALUES
  -- Anthropic. Cache reads bill at 0.1x input, cache writes at 1.25x.
  ('claude-opus-5',    'anthropic', 'Claude Opus 5',   'claude', 200000, 64000, 15.00, 75.00, 1.50, 18.75, 1, 1, 1, 'active', '2026-09-06T00:00:00Z', '2026-09-06T00:00:00Z'),
  ('claude-sonnet-5',  'anthropic', 'Claude Sonnet 5', 'claude', 200000, 64000,  3.00, 15.00, 0.30,  3.75, 1, 1, 0, 'active', '2026-09-06T00:00:00Z', '2026-09-06T00:00:00Z'),
  ('claude-haiku-4-5-20251001', 'anthropic', 'Claude Haiku 4.5', 'claude', 200000, 32000, 1.00, 5.00, 0.10, 1.25, 0, 1, 0, 'active', '2026-09-06T00:00:00Z', '2026-09-06T00:00:00Z'),
  -- OpenAI.
  ('gpt-5.1',          'openai', 'GPT-5.1',       'gpt', 400000, 128000, 1.25, 10.00, 0.125, 0, 1, 1, 1, 'active', '2026-09-06T00:00:00Z', '2026-09-06T00:00:00Z'),
  ('gpt-5.1-codex',    'openai', 'GPT-5.1 Codex', 'gpt', 400000, 128000, 1.25, 10.00, 0.125, 0, 1, 1, 0, 'active', '2026-09-06T00:00:00Z', '2026-09-06T00:00:00Z'),
  -- The CLI providers report a model name but bill through the seat, not per token; these
  -- rows exist so a subscription usage_record can still name what answered it.
  ('codex-cli-default',  'codex',  'Codex CLI (subscription)',  'gpt',    400000, 128000, 0, 0, 0, 0, 1, 1, 1, 'active', '2026-09-06T00:00:00Z', '2026-09-06T00:00:00Z'),
  ('claude-cli-default', 'claude', 'Claude CLI (subscription)', 'claude', 200000,  64000, 0, 0, 0, 0, 1, 1, 1, 'active', '2026-09-06T00:00:00Z', '2026-09-06T00:00:00Z'),
  ('cursor-cli-default', 'cursor', 'Cursor Agent (subscription)', 'mixed', 200000, 64000, 0, 0, 0, 0, 0, 0, 1, 'active', '2026-09-06T00:00:00Z', '2026-09-06T00:00:00Z')
ON CONFLICT (id) DO UPDATE SET
  provider_id = excluded.provider_id,
  display_name = excluded.display_name,
  family = excluded.family,
  context_window_tokens = excluded.context_window_tokens,
  max_output_tokens = excluded.max_output_tokens,
  price_input_per_mtok = excluded.price_input_per_mtok,
  price_output_per_mtok = excluded.price_output_per_mtok,
  price_cache_read_per_mtok = excluded.price_cache_read_per_mtok,
  price_cache_write_per_mtok = excluded.price_cache_write_per_mtok,
  supports_reasoning = excluded.supports_reasoning,
  supports_vision = excluded.supports_vision,
  is_default = excluded.is_default,
  status = excluded.status,
  updated_at = excluded.updated_at;
