# Project Rules for Antigravity Agent

## Enforced Custom Plugins
- **Ponytail Agent Plugin (YAGNI)**: Enforce a minimalist, pragmatic coding style. Always prioritize standard library features, native functions, and reuse of existing helpers. Before writing any code, apply the decision ladder:
  1. Question existence (YAGNI).
  2. Reuse codebase solutions.
  3. Use standard libraries.
  4. Use native features.
  5. Use existing dependencies.
  6. Minimize to one-liner if possible.
  7. Write the minimum code required.
  Do not introduce complex, nested wrapper abstractions unless requested or mathematically essential. Never cut corners on safety, error handling, or performance correctness.
