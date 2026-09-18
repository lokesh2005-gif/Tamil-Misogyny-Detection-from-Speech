"""
Centralized API Key Manager for Multi-Key Pools.
Supports round-robin rotation, automatic failover on rate-limiting/quota exhaustion,
and zero key leakage in logs or user interfaces.
"""

import os
import time
import threading
from typing import Dict, List, Any, Optional, Callable, Tuple
from dotenv import load_dotenv

load_dotenv()


class APIKeyManager:
    """
    Thread-safe Centralized API Key Manager supporting multiple keys per provider.

    Key Features:
    1. Loads all configured keys at startup from environment variables.
    2. Keeps keys exclusively in memory.
    3. Round-Robin scheduling: KEY 1 -> KEY 2 -> ... -> KEY N -> KEY 1.
    4. Automatic failover on rate-limit / quota errors (429, ResourceExhausted).
    5. Temporary key cooldown on quota exhaustion (skips until cooldown expires).
    6. Finite retries bounded by number of available keys (never retries indefinitely).
    7. Complete key masking: prevents leaking raw credentials in errors, logs, or UI.
    """

    def __init__(self, cooldown_seconds: float = 60.0):
        self.cooldown_seconds = cooldown_seconds
        self._lock = threading.Lock()
        self._pools: Dict[str, List[Dict[str, Any]]] = {}
        self._rr_index: Dict[str, int] = {}
        self._known_secrets: set = set()
        self.reload_keys()

    def reload_keys(self, load_env_file: bool = True) -> None:
        """Scan environment variables and populate provider pools."""
        with self._lock:
            if load_env_file:
                load_dotenv(override=True)
            self._pools.clear()
            self._known_secrets.clear()

            # Discover Gemini keys
            gemini_keys = self._discover_keys_for_provider(
                prefix="GEMINI_API_KEY",
                fallback_env="GEMINI_API_KEY",
                generic_prefixes=["API_KEY"]
            )
            self._init_pool("gemini", gemini_keys)

            # Discover ElevenLabs keys
            eleven_keys = self._discover_keys_for_provider(
                prefix="ELEVENLABS_API_KEY",
                fallback_env="ELEVENLABS_API_KEY",
                generic_prefixes=[]
            )
            self._init_pool("elevenlabs", eleven_keys)

    def _discover_keys_for_provider(
        self,
        prefix: str,
        fallback_env: str,
        generic_prefixes: List[str],
    ) -> List[str]:
        keys = []
        seen = set()

        def add_key(val: Optional[str]):
            if not val:
                return
            cleaned = str(val).strip()
            # Ignore empty strings or common placeholders
            if not cleaned or cleaned.startswith("your_") or cleaned.startswith("your-"):
                return
            if cleaned not in seen:
                seen.add(cleaned)
                keys.append(cleaned)
                self._known_secrets.add(cleaned)

        # 1. Check indexed keys: PREFIX_1, PREFIX_2, ..., PREFIX_50
        for i in range(1, 51):
            var_name = f"{prefix}_{i}"
            if var_name in os.environ:
                add_key(os.getenv(var_name))

        # 2. Check fallback single env
        if fallback_env in os.environ:
            add_key(os.getenv(fallback_env))

        # 3. Check generic prefixes: API_KEY_1, API_KEY_2, ...
        for gen in generic_prefixes:
            for i in range(1, 51):
                var_name = f"{gen}_{i}"
                if var_name in os.environ:
                    add_key(os.getenv(var_name))
            if gen in os.environ:
                add_key(os.getenv(gen))

        return keys

    def _init_pool(self, provider: str, keys: List[str]) -> None:
        provider = provider.lower()
        self._pools[provider] = []
        for i, k in enumerate(keys):
            self._pools[provider].append({
                "key": k,
                "alias": f"Key-{i+1}",
                "index": i,
                "cooldown_until": 0.0,
                "failure_count": 0,
                "success_count": 0,
                "last_used": 0.0,
            })
        self._rr_index[provider] = 0

    def has_keys(self, provider: str = "gemini") -> bool:
        provider = provider.lower()
        with self._lock:
            return bool(self._pools.get(provider))

    def get_key_count(self, provider: str = "gemini") -> int:
        provider = provider.lower()
        with self._lock:
            return len(self._pools.get(provider, []))

    def get_next_key(self, provider: str = "gemini") -> Tuple[Optional[str], Optional[str]]:
        """
        Select next available key using Round-Robin.
        Skips keys currently cooling down from quota/rate-limits.
        Returns (api_key, key_alias) or (None, None).
        """
        provider = provider.lower()
        with self._lock:
            pool = self._pools.get(provider, [])
            if not pool:
                return None, None

            now = time.time()
            n = len(pool)
            start_idx = self._rr_index.get(provider, 0) % n

            # Search for first non-cooling key starting from current RR pointer
            for step in range(n):
                idx = (start_idx + step) % n
                entry = pool[idx]
                if entry["cooldown_until"] <= now:
                    entry["cooldown_until"] = 0.0
                    entry["last_used"] = now
                    self._rr_index[provider] = (idx + 1) % n
                    return entry["key"], entry["alias"]

            # If all keys are in cooldown, pick the one that will expire soonest
            best_idx = min(range(n), key=lambda i: pool[i]["cooldown_until"])
            entry = pool[best_idx]
            entry["last_used"] = now
            self._rr_index[provider] = (best_idx + 1) % n
            return entry["key"], entry["alias"]

    def mark_rate_limited(self, provider: str, key: str, reason: str = "Rate limit / Quota exceeded") -> None:
        """Temporarily put a key into cooldown without removing it."""
        provider = provider.lower()
        with self._lock:
            pool = self._pools.get(provider, [])
            for entry in pool:
                if entry["key"] == key:
                    entry["cooldown_until"] = time.time() + self.cooldown_seconds
                    entry["failure_count"] += 1
                    break

    def mark_success(self, provider: str, key: str) -> None:
        provider = provider.lower()
        with self._lock:
            pool = self._pools.get(provider, [])
            for entry in pool:
                if entry["key"] == key:
                    entry["success_count"] += 1
                    entry["failure_count"] = 0
                    entry["cooldown_until"] = 0.0
                    break

    def execute_with_failover(
        self,
        provider: str,
        operation: Callable[[str], Any],
        is_rate_limit_fn: Optional[Callable[[Exception], bool]] = None,
        max_attempts: Optional[int] = None,
    ) -> Any:
        """
        Execute an API call using round-robin keys.
        Automatically catches rate-limit/quota errors, marks the key for cooldown,
        and fails over to the next available key.
        Never retries indefinitely (at most len(pool) attempts).
        """
        provider = provider.lower()
        total_keys = self.get_key_count(provider)
        if total_keys == 0:
            raise RuntimeError(f"No API keys configured for provider '{provider}'.")

        attempts = max_attempts if max_attempts is not None else total_keys
        attempts = max(1, min(attempts, total_keys))

        last_exception = None

        for attempt in range(attempts):
            api_key, alias = self.get_next_key(provider)
            if not api_key:
                raise RuntimeError(f"No usable API key available for provider '{provider}'.")

            try:
                result = operation(api_key)
                self.mark_success(provider, api_key)
                return result
            except Exception as ex:
                last_exception = ex
                is_rate_limited = False

                if is_rate_limit_fn:
                    is_rate_limited = is_rate_limit_fn(ex)
                else:
                    is_rate_limited = is_rate_limit_or_quota_error(ex)

                if is_rate_limited:
                    self.mark_rate_limited(provider, api_key, str(ex))
                    continue
                else:
                    clean_msg = self.sanitize_text(str(ex))
                    raise RuntimeError(clean_msg) from None

        clean_err = self.sanitize_text(str(last_exception)) if last_exception else "All configured API keys exhausted."
        raise RuntimeError(f"All {total_keys} configured API keys for '{provider}' failed. Last error: {clean_err}")

    def get_pool_status(self, provider: str = "gemini") -> Dict[str, Any]:
        """
        Return sanitized telemetry for UI/Monitoring.
        Never returns secret key strings.
        """
        provider = provider.lower()
        now = time.time()
        with self._lock:
            pool = self._pools.get(provider, [])
            total = len(pool)
            active = sum(1 for e in pool if e["cooldown_until"] <= now)
            cooling = total - active
            curr_idx = self._rr_index.get(provider, 0) % total if total > 0 else 0

            return {
                "provider": provider,
                "total_keys": total,
                "active_keys": active,
                "cooling_keys": cooling,
                "current_index": curr_idx + 1 if total > 0 else 0,
                "has_failover": total > 1,
            }

    def sanitize_text(self, text: str) -> str:
        """Replace any known raw secret with [REDACTED_API_KEY]."""
        if not text:
            return text
        sanitized = str(text)
        for secret in self._known_secrets:
            if secret and secret in sanitized:
                sanitized = sanitized.replace(secret, "[REDACTED_API_KEY]")
        return sanitized


def is_rate_limit_or_quota_error(exception: Exception) -> bool:
    """
    Detect rate limit, quota, resource exhaustion, or billing issues
    across Google GenAI and ElevenLabs APIs.
    """
    msg = str(exception).lower()
    quota_cues = [
        "429",
        "resource_exhausted",
        "resourceexhausted",
        "quota",
        "rate limit",
        "ratelimit",
        "too many requests",
        "credits exceeded",
        "exceeded your current quota",
        "insufficient_quota",
        "usage_limit_reached",
        "capacity_exceeded",
    ]
    return any(cue in msg for cue in quota_cues)


# Centralized global singleton instance
key_manager = APIKeyManager()
