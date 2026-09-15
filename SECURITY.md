# Security policy

Lune Voice handles dictated speech and the text it turns into, so security
reports are welcome and taken seriously.

## Reporting a vulnerability

Report it privately through GitHub: open the repository's **Security** tab and
choose **Report a vulnerability**, or go straight to
<https://github.com/CoderDayton/lune-voice/security/advisories/new>.

Please do not open a public issue for a vulnerability.

A useful report says:

- what the problem is and what an attacker gains from it
- how to reproduce it
- the Lune Voice version (Settings → General shows it) and your OS

## Supported versions

Security fixes go into the latest release only.

## Scope

In scope, for example:

- audio or transcript text leaving the machine without the user choosing a
  remote backend, or `privacy_mode` being bypassed
- transcripts, API keys or settings readable by another local user
- privilege escalation through the udev rule or the `pkexec` prompts
- a way past the signature check on updates or the SHA-256 pin on the GPU
  plugin and model downloads

Out of scope: problems that need an attacker who already controls your user
account, and bugs in a remote styling or transcription endpoint you configured
yourself.
