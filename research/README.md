# Research

This directory contains non-authoritative mathematical research records that are intentionally separated from `mission/QPS` federation material.

Current controlled promotion: W008-W011 recursive-geometry/PCA research. See `W008_W011_RESEARCH_REGISTER.md` and `W008_W011_PROMOTION_MANIFEST.json`.

Method application cards: `MATH_METHOD_APPLICATION_CARDS_v1.yaml` defines how a consumer records where and how a method card is applied (data, thresholds, window, inference rules, narrative and verdict). It adds no vocabulary: edge types stay with `MATH_METHOD_DEPENDENCY_GRAPH_v1.yaml` and threshold kinds with `MATH_METHOD_INFO_CARDS_v1.yaml`. A synthetic example is in `examples/method_application_card_synthetic.yaml`. Filled cards with project data stay with the consumer.

Method references: `MATH_METHOD_REFERENCES_v1.yaml` binds public mathematical references to existing card IDs, kernels, challenge tests, assumptions and limits. The governed release build generates `methods/reference-library.html` and links it from the method navigator. References support understanding; executable qualification still requires the method's tests and receipts. The four primary-paper DOI titles were checked against Crossref on 2026-11-12; documentation links are provided as implementation references.

Depth programme (#110): the current governed build already contains the method navigator. The reference library addresses #133 and complements the application schema merged in #108. The original October 22 review has not been recovered in this workspace or the connected Drive searches, so its private-content screening and original W5b–W9 scopes remain unresolved. This library does not reconstruct or claim completion of that missing review.
