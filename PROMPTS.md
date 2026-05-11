# Development Prompts Log

This document captures the key engineering steps taken during this session to build and secure the Upstox Trading Dashboard.

### 1. Security & Architecture Hardening
**Accomplished**: Integrated `cryptography.fernet` for data-at-rest encryption of tokens and implemented root-path CSRF state validation for OAuth flows. Added log sanitization to redact sensitive authorization codes.

### 2. Resolving OAuth State Mismatches
**Accomplished**: Diagnosed and fixed the "Invalid state parameter" error by forcing explicit `path="/"` on security cookies and ensuring consistency between `localhost` and `127.0.0.1` contexts.

### 3. Database Initialization & Path Safety
**Accomplished**: Fixed `sqlite3.OperationalError` by implementing absolute path resolution for the database file inside Docker volumes and ensuring robust table creation during the FastAPI lifespan startup.

### 4. SSE Stream Concurrency & Thread Safety
**Accomplished**: Resolved the "empty dashboard" issue by converting `broadcast_tick` into a thread-safe synchronous function, allowing the background websocket streamer to correctly push updates to the FastAPI event loop.
