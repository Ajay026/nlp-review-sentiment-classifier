# Original ProjectPro source: comparison only

The files under `projectpro/` are byte-for-byte copies of the inspected modular
ZIP's Python sources, README and requirements. Keep their ProjectPro attribution.
They are **not the default runnable pipeline** and have not been executed here.

Known issues: feature vectorizers are fitted before train/test splitting;
requirements target an older Python environment; NLTK resources are needed;
prediction loads pickle files and should never be pointed at untrusted artifacts.
Do not install the reference requirements into the pilot environment or load the
original pretrained binaries. The root `Engine.py`, `predict.py` and `Source/`
provide the corrected, tested implementation.

Original notebooks (including their outputs), dataset, PDF, caches and pretrained
pickle files are preserved privately but excluded here. Dataset metadata contains
reviewer identifiers. The methodology PDF's redistribution permission remains
unresolved; a newly written summary appears in `docs/METHODOLOGY.md`.

ProjectPro's official demo advertises pushing downloaded project code and datasets
to a learner's GitHub portfolio: https://www.projectpro.io/project/project-demo?source=homeDemo
That specific portfolio guidance is not a blanket open-source license. No
LICENSE/NOTICE file was found in the supplied archive. No new license is asserted
for the original source, nor authorship claimed on behalf of Ajay.
