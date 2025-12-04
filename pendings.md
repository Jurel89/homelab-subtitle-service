- Add github actions workflows to:
    - On PR from braches like feature-** to main, run tests, software quality, security etc, like we've already done in the security-quality.yml
    - On merge to main, run tests again, software quality, security, build a docker image in CI, optionally deploy to a self-hosted runner
    - Only deploy to docker hub when a new git tag is created, so we leave docker hub clean with only the stable and relevant upgrades

- Add a functionality so this tool supports the translation to another language other than english, which is not supported by fast-whisper

- Add a functionality so, when you already have a human generated subtitle, you can trigger a syncronization, so that subtitled is modified to match exactly your video file

- Add a functionality so, when you already have a human generated subtitle, and you trigger the automatic transcription with faster-whisper, there's a precision/quality comparison between the two. The tool should auto-discover that subtitle

- Add a functionality to store the logs of the application, and add to those logs the memory, CPU%, GPU% etc.

- Add a web UI, where we can trigger new subtitle generations, explore folders and files within a file-system, have a history page with all the jobs and their progress, explore logs, etc.

- Add a tutorial, features and model accuracy and logic explanation as a wiki in the web UI

- Generate an API backend, so we can easily integrate this with other tools/services

- Add a job queue service so we can trigger new jobs, or even batches, and they await until the tool has resources to execute them

🧭 Recommended Execution Order (for maximum momentum)
✔ CI for PR → main → tag-based release
✔ Logging & performance metrics
Add translation
Add subtitle syncing
Add accuracy comparison
Add job queue & structured logs
Add FastAPI API backend
Add SQLite job persistence
Add Web UI (HTMX or React)
Add wiki/tutorial inside UI