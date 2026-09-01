# Repository knowledge lifecycle

Repository knowledge uses freshness-first semantics: before serving a search, the system verifies that the prepared index still represents the current repository and rebuilds it when the source has changed. A rebuilt index becomes active only after preparation succeeds; if initial preparation or refresh fails, repository search fails closed rather than serving stale or partially prepared evidence, because repository analysis should prefer authoritative source state over availability.
