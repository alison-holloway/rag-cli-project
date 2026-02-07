# Security Guide

This document covers security practices for the RAG CLI project, particularly around API key handling.

## Security Audit Summary

**Date:** 2026-02-07
**Status:** Passed with recommendations

### Verified Safe

| Area | Status | Notes |
|------|--------|-------|
| API key storage | SAFE | Only read from environment variables via Pydantic |
| API key logging | SAFE | Never logged or printed |
| API key serialization | SAFE | Not included in `/api/config` response |
| Anthropic SDK usage | SAFE | Key passed directly to `anthropic.Anthropic()` |
| .gitignore coverage | SAFE | Covers `.env`, `.env.*`, credentials |
| .env.example | SAFE | Only placeholders, no real keys |
| Git history | SAFE | No credential files ever committed |
| Frontend | SAFE | No API keys handled client-side |

### Fixed Issues

| Severity | Issue | Resolution |
|----------|-------|------------|
| MEDIUM | File paths in error messages | Added error sanitization |
| MEDIUM | Raw exceptions in API responses | Sanitized before returning |

---

## API Key Handling

### How Keys Are Loaded

API keys are loaded exclusively from environment variables using Pydantic BaseSettings:

```python
# src/config.py
class LLMSettings(BaseModel):
    anthropic_api_key: str | None = Field(
        default=None,
        description="Anthropic API key for Claude",
    )
```

The key is:
- Never logged (all log statements checked)
- Never serialized to JSON/YAML
- Never included in API responses
- Passed directly to the Anthropic SDK

### Setting Up Your API Key

1. **Copy the example file:**
   ```bash
   cp .env.example .env
   ```

2. **Add your API key:**
   ```bash
   # In .env file
   ANTHROPIC_API_KEY=sk-ant-your-key-here
   ```

3. **Verify .env is ignored:**
   ```bash
   git status  # .env should NOT appear
   ```

### Never Do This

```python
# BAD: Logging the key
logger.info(f"Using API key: {settings.llm.anthropic_api_key}")

# BAD: Including in API response
return {"config": settings.model_dump()}

# BAD: Printing to console
print(f"Key: {api_key}")

# BAD: Hardcoding
api_key = "sk-ant-abc123..."
```

---

## Pre-Commit Security Checklist

Before committing code, verify:

- [ ] **No API keys in code:**
  ```bash
  git diff --cached | grep -iE "sk-|api_key\s*=\s*['\"][^'\"]+['\"]"
  ```

- [ ] **.env not staged:**
  ```bash
  git status | grep -E "\.env$" && echo "WARNING: .env staged!"
  ```

- [ ] **No config dumps:**
  ```bash
  git diff --cached | grep -E "print.*settings|log.*config"
  ```

- [ ] **No hardcoded secrets:**
  ```bash
  git ls-files | xargs grep -l "sk-ant-\|sk-[a-zA-Z0-9]{20,}" 2>/dev/null
  ```

### Quick Security Check

Run this before pushing:

```bash
# Check for any potential secrets in staged files
git diff --cached --name-only | xargs grep -lE \
  "api_key|secret|password|token" 2>/dev/null | \
  xargs grep -nE "=\s*['\"][^'\"]{10,}['\"]" 2>/dev/null
```

---

## Safe Testing with API Keys

### Local Development

1. **Use environment variables:**
   ```bash
   export ANTHROPIC_API_KEY=sk-ant-your-key
   rag-cli query "test" --llm claude
   ```

2. **Or use .env file:**
   ```bash
   # .env (git-ignored)
   ANTHROPIC_API_KEY=sk-ant-your-key
   ```

3. **Verify key is loaded:**
   ```bash
   rag-cli config list | grep -i anthropic
   # Should show "anthropic_api_key: [SET]" not the actual key
   ```

### CI/CD Environments

- Use GitHub Secrets or equivalent
- Never echo or print secrets in logs
- Set `ANTHROPIC_API_KEY` as a secret environment variable

### Testing Without Real Keys

For unit tests, use mock keys:

```python
# tests/test_llm_client.py
def test_claude_client():
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
        client = ClaudeClient()
        # Test with mocked responses
```

---

## Error Message Security

API error responses are sanitized to remove sensitive paths:

```python
# Before: "File not found: /Users/alison/project/data/secret.pdf"
# After:  "File not found: secret.pdf"
```

This prevents:
- Exposure of filesystem structure
- Leaking of usernames from paths
- Revealing internal project organization

---

## If You Accidentally Commit a Key

### Immediate Steps

1. **Revoke the key immediately:**
   - Go to https://console.anthropic.com
   - Delete the compromised API key
   - Generate a new one

2. **Remove from git history:**
   ```bash
   # Install git-filter-repo if needed
   pip install git-filter-repo

   # Remove the file from all commits
   git filter-repo --path .env --invert-paths

   # Or use BFG Repo-Cleaner
   bfg --delete-files .env
   ```

3. **Force push (coordinate with team):**
   ```bash
   git push --force-with-lease
   ```

4. **Notify team members** to re-clone the repository

### Prevention

- Use pre-commit hooks to catch secrets
- Consider tools like `git-secrets` or `truffleHog`
- Review diffs before committing

---

## Security Contacts

If you discover a security vulnerability:

1. Do not open a public issue
2. Contact the maintainers directly
3. Allow time for a fix before disclosure

---

## Changelog

| Date | Change |
|------|--------|
| 2026-02-07 | Initial security audit completed |
| 2026-02-07 | Added error sanitization for API responses |
