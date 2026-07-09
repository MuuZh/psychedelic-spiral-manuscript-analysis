# Manuscript Analysis Scripts

This directory contains manuscript-specific analysis scripts copied from the analysis workspace. The scripts cover all-metrics summaries, DFR analyses, phase/FC group analyses, added GIPR and CAI-related analyses, path entropy, MSD beta, phase-gradient alignment, GCOR, and NGSC paired analyses.

`upstream_generation/` contains manuscript-specific intermediate-generation scripts retained to document how downstream inputs were produced, including phase-field reconstruction, empirical and reconstructed phase-FC, per-pattern MSD beta, GCOR batch outputs, sigma/curl sensitivity inputs, and curl-vs-optical-flow comparison inputs.

`raw_fc/` contains the conventional raw-FC pipeline and analysis scripts found in the local FC workspace, including the hemisphere-first comparison script referenced by the retained raw-FC export workflow.

These files are archived for transparency. They generally expect derived fMRI inputs and prior analysis outputs that are not distributed in this repository. Local paths have been sanitized where found, but the scripts have not been revalidated end-to-end after public-release cleanup.
