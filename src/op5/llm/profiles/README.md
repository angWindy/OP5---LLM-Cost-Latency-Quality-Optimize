# OP5 LLM Judge — Profile Directory
#
# Each YAML file defines ONE provider's chain of models.
# Format:
#   base_url  : OpenAI-compatible endpoint (required)
#   api_key   : API key (required; use env var interpolation: "${ENV_VAR}")
#   api_type  : "openai" | "nvidia" | "gemini" (default: auto-detect from base_url)
#   models    : ordered list of model IDs to try in sequence
#   timeout   : per-call timeout in seconds (default: 60)
#   temperature: sampling temperature (default: 0.1)
#   max_tokens: max completion tokens (default: 800)
#   label     : human-readable short name (default: filename without .yaml)
#
# Env var interpolation: "${VAR_NAME}" in any field is replaced with os.environ[VAR_NAME].
# Sensitive values should reference env vars, never be hardcoded here.
#
# Chaining: the LLMJudge router tries profiles in order until one succeeds.
# See config.yaml for the default chain order.
