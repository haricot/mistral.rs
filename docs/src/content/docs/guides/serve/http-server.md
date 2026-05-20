---
title: Configure the HTTP server
description: Host, port, body limits, and authentication for mistralrs serve.
sidebar:
  order: 1
---

`mistralrs serve` defaults: bind on `0.0.0.0:1234`, CORS permissive (any origin), no authentication, 50 MB request body limit.

## Host and port

```bash
mistralrs serve --host 127.0.0.1 --port 8080 -m <model>
```

`--host` controls the bind interface. `0.0.0.0` (default) accepts connections from any host on the network; `127.0.0.1` restricts to the local machine. `--port` is the TCP port (default 1234).

## CORS and body limit

By default, the server allows cross-origin requests from any origin. You can restrict this using the `--cors-origins` flag:

```bash
mistralrs serve --cors-origins http://localhost:3000,https://app.example.com -m <model>
```

For advanced server configuration (CORS, Swagger, body limits), you can use a TOML file with `--config-srv`:

```bash
mistralrs serve --config-srv server-config.toml -m <model>
```

The default body limit is 50 MB. Allowed methods are `GET`, `POST`, `PUT`, `DELETE`; allowed headers include `Content-Type` and `Authorization`.

## Authentication

mistral.rs does not implement authentication. The intended pattern is a reverse proxy (nginx, Caddy, Traefik) handling authentication and TLS.

OpenAI-protocol clients always send an `Authorization: Bearer ...` header because the OpenAI SDK requires an API key at initialization. mistral.rs does not validate the header.

## Logging

```bash
mistralrs serve -v -m <model>
```

Use `-vv` for trace-level file/cache internals. Module filters are still available with `RUST_LOG`, for example `RUST_LOG=mistralrs_core=debug,tower_http=info`.

## Config file versus flags

Most CLI flags have a TOML config equivalent. Run with `mistralrs from-config -f config.toml`.

Full schema: [CLI TOML config reference](/mistral.rs/reference/cli-toml-config/). Minimal example:

```toml
command = "serve"

[server]
host = "127.0.0.1"
port = 8080

[runtime]
enable_search = true
enable_code_execution = true

[[models]]
kind = "text"
model_id = "Qwen/Qwen3-4B"

[models.quantization]
in_situ_quant = "4"
```

## See also

- [Agentic runtime for apps](/mistral.rs/guides/agents/agentic-runtime/).
- [Running multiple models](/mistral.rs/guides/serve/multiple-models/).
- [Production checklist](/mistral.rs/guides/deploy/production-checklist/).
