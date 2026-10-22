---
paths:
  - "**/Containerfile"
  - "compose.yaml"
  - "config/**"
  - "scripts/**"
  - ".github/**"
  - "docs/operations/**"
---
<!-- Written by kiln and kept current by `kiln update`: do not edit, it would be overwritten. -->
# Delivery and operations

- **[FAST-CI]** The full CI runs in under 10 minutes; a flaky test is quarantined and fixed within
  the week, never re-run until it passes.
- **[IMAGES-BY-DIGEST]** Every container base image is pinned by its `@sha256:` digest, never by
  a moving tag alone.
- **[HARDENED-CONTAINERS]** Rootless containers and a non-root user, a read-only file system
  outside declared volumes, `cap_drop: ALL`, no secret in the image nor in its layers.
- **[HEALTH-PROBES]** Each service exposes a liveness probe and a readiness probe; the second one
  really checks its dependencies (database, cache).
