# Security boundary

YoloWebAgentStarter is local, single-user software. It has no authentication, authorization, TLS, or tenant separation. Keep the default `127.0.0.1` bind address and do not expose the API or frontend directly to a network.

Runtime files are constrained to `YWA_DATA_DIR`; directory scans are constrained to `YWA_IMPORT_ROOT`; image, run, export, and model helpers reject paths escaping their managed roots. Local model paths used for training are accepted only from the managed model directory. These checks reduce accidental path traversal but do not turn Starter into a hardened multi-user service.

## Private vulnerability reports

Before reporting a security issue, remove secrets, private datasets, model weights, and customer data from the reproduction.

**Current status: release-blocking, not configured.** The maintainer must enable GitHub private vulnerability reporting for this repository, or add a maintainer-controlled security email, before the first public release. Once GitHub reporting is enabled, the private report entry point is:

<https://github.com/fengzhizi715/YoloWebAgentStarter/security/advisories/new>

Until that setting is enabled, do not disclose vulnerabilities in a public issue. The release checklist requires the maintainer to verify that this link opens a private report form (or to replace it with the approved security mailbox) before publishing a tag.
