# Phase 6 retrieval experiments (30 answerable dev questions, k=4)

| Experiment | Change | Chunks | Hit@1 | Hit@4 | MRR | Δ MRR | Ctx precision | Ctx recall | Avg context chars | ms/query | Missed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | vector, all-MiniLM-L6-v2, chunk 800 | 305 | 0.767 | 0.967 | 0.867 | +0.000 | 0.850 | 0.967 | 2449 | 14 | a11 |
| chunk400 | vector, all-MiniLM-L6-v2, chunk 400 | 578 | 0.867 | 0.933 | 0.900 | +0.033 | 0.863 | 0.933 | 1172 | 13 | a02, a11 |
| chunk1200 | vector, all-MiniLM-L6-v2, chunk 1200 | 204 | 0.800 | 0.967 | 0.875 | +0.008 | 0.852 | 0.967 | 3682 | 13 | a11 |
| emb_bge_small | vector, bge-small-en-v1.5, chunk 800 | 305 | 0.867 | 0.967 | 0.917 | +0.050 | 0.897 | 0.967 | 2536 | 36 | a11 |
| emb_multi_minilm | vector, paraphrase-multilingual-MiniLM-L12-v2, chunk 800 | 305 | 0.733 | 0.900 | 0.806 | -0.061 | 0.783 | 0.900 | 2415 | 27 | a09, a11, a19 |
| emb_multi_e5 | vector, multilingual-e5-small, chunk 800 | 305 | 0.900 | 1.000 | 0.950 | +0.083 | 0.914 | 1.000 | 2490 | 40 | - |
| bm25 | bm25, chunk 800 | 305 | 0.833 | 0.967 | 0.900 | +0.033 | 0.867 | 0.950 | 2558 | 1 | a06 |
| hybrid | hybrid, all-MiniLM-L6-v2, chunk 800 | 305 | 0.867 | 1.000 | 0.928 | +0.061 | 0.892 | 1.000 | 2511 | 20 | - |
| rerank | vector + rerank, all-MiniLM-L6-v2, chunk 800 | 305 | 0.900 | 1.000 | 0.944 | +0.078 | 0.918 | 1.000 | 2503 | 1048 | - |
| hybrid_rerank | hybrid + rerank, all-MiniLM-L6-v2, chunk 800 | 305 | 0.900 | 1.000 | 0.944 | +0.078 | 0.918 | 1.000 | 2501 | 1086 | - |
| hybrid_rerank_bge | hybrid + rerank, bge-small-en-v1.5, chunk 800 | 305 | 0.900 | 1.000 | 0.944 | +0.078 | 0.918 | 1.000 | 2498 | 1040 | - |
| hybrid_e5 | hybrid, multilingual-e5-small, chunk 800 | 305 | 0.867 | 0.967 | 0.917 | +0.050 | 0.883 | 0.967 | 2530 | 28 | a06 |
| rerank_e5 | vector + rerank, multilingual-e5-small, chunk 800 | 305 | 0.900 | 1.000 | 0.944 | +0.078 | 0.918 | 1.000 | 2496 | 1729 | - |
