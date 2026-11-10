# GPU training

Launch prepared direct GPU training through
`scripts/sync_gpu_dashboard.py --launch-command <remote-command.json>` with the
run's connection, container, output and job arguments documented in README.md.
This launch path automatically starts persistent dashboard mirroring every
20 seconds. Do not start a bare trainer and leave dashboard activation as a
separate manual step, or substitute a five-minute monitoring cron for this
mirror. Run the launcher as a persistent background process and verify its
first successful refresh. Reconcile an existing trainer before retrying;
never restart or duplicate it just to attach the dashboard.
