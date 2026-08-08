# Security boundary

YoloWebAgentStarter is local, single-user software. It has no authentication, authorization, TLS, or tenant separation. Keep the default `127.0.0.1` bind address and do not expose the API or frontend directly to a network.

Runtime files are constrained to `YWA_DATA_DIR`; directory scans are constrained to `YWA_IMPORT_ROOT`; image, run, export, and model helpers reject paths escaping their managed roots. Local model paths used for training are accepted only from the managed model directory. These checks reduce accidental path traversal but do not turn Starter into a hardened multi-user service.

Before reporting a security issue, remove secrets, private datasets, model weights, and customer data from the reproduction. The repository has no private security contact yet; maintainers must publish one before claiming a public vulnerability disclosure process.
