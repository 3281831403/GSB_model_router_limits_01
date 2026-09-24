# Security Policy

## Supported versions

Only the latest release receives security fixes.

## Reporting a vulnerability

Please **do not** open a public issue. Report it privately through [GitHub security advisories](https://github.com/TheCoder30ec4/model_router_python/security/advisories/new). You can expect a first response within 7 days.

## How this library handles your keys

- Keys are passed in code and kept only in memory. The library never reads `.env` files or environment variables, and never writes keys to disk or logs.
- Only the routing key (`jev_api_key` or `openrouter_api_key`) is sent over the network, over HTTPS, to jevai.org or openrouter.ai respectively.
- Provider keys passed in `providers={...}` are **never sent anywhere**. They are only returned to you by `router.api_key_for()`.
- The first 8,000 characters of each task are sent to the routing service. Don't route prompts containing data you aren't allowed to share with Jev / OpenRouter.
