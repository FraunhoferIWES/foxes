# Data Classification Policy

Use AI tools only for data classified `public` or `restricted`. Do not send
`confidential` or `strictly confidential` content to an AI tool unless the user
explicitly releases that exact scope under
[the exception process](data-classification-exceptions.md).

| Class | Definition |
|---|---|
| `public` | Intended for the general public, with no damage if shared. |
| `restricted` | For employees and other authorized persons; disclosure or loss has limited, manageable impact. |
| `confidential` | Disclosure or loss could cause considerable damage, including certain project or customer information. |
| `strictly confidential` | Disclosure or loss could be existentially threatening or have legal consequences, including specially protected personal data. |

Classification follows the data, not the file it sits in. This applies to
prompts, attachments, fixtures, logs, screenshots and files an agent reads.

Ask the user for classification before processing:

- personal data beyond work contacts;
- customer, partner, contract, pricing or funding data;
- unpublished, patent-relevant or NDA-covered research data;
- credentials, keys or unremediated vulnerabilities; or
- data of unknown provenance.

Name the data in question and offer a way to proceed without exposing it, such
as synthetic or anonymized samples, structure without real content, or a step
the user performs without an AI tool. Never guess a class, silently redact, or
process only the parts that seem harmless. The explicitly identified security
fixture is exempt from the vulnerability-classification prompt.

An explicit release covers only the named scope. Record it in
[data-classification-exceptions.md](data-classification-exceptions.md) in the
same change and name that entry in the completion report. If the software
itself will process the released data, record the resulting boundary in
[architecture.md](architecture.md#data-and-integration-boundaries).
