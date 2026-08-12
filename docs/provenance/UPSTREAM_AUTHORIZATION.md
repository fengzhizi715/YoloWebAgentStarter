# Upstream authorization release gate

```text
Release status: pending
Upstream commit: 701f6e5a63b73f39e35f48fb6de7d2414401875a
Rights holder:
Approved date:
Evidence reference:
```

Public release is blocked.

The recorded YoloWebAgent baseline is commit `701f6e5a63b73f39e35f48fb6de7d2414401875a` from `https://github.com/fengzhizi715/YoloWebAgent.git`. At capture, its top-level tree had no `LICENSE` or `NOTICE`. A local MIT file cannot create a right to relicense code from that tree.

Before a public tag, the copyright holder or an authorized representative must replace this template with verifiable evidence containing all of the following:

1. `Status: approved`.
2. The exact commit SHA above (or a separately approved replacement SHA).
3. The applicable upstream license text or a written grant that explicitly permits the selective derivative and its intended public license.
4. The rights holder/authorized representative, date, and a durable reference (signed document, public license URL, or archived correspondence identifier).
5. A confirmation that the file-level areas in [`../../migration_matrix.md`](../../migration_matrix.md) are covered, or a list of excluded/re-written files.

Do not put private correspondence, signatures, customer data, or credentials in the public repository. Store the underlying evidence in the maintainer-controlled records system and place only a non-sensitive durable reference here. `scripts/check_release_provenance.py` blocks a tag CI run until this document is marked approved with the required fields.
