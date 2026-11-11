# baseline vs final

- baseline: vector[all-MiniLM-L6-v2]
- final: vector[multilingual-e5-small]
- generator: groq/openai/gpt-oss-20b, prompt v2
- judge: openai/gpt-oss-120b

| Metric | baseline | final | Change |
|---|---|---|---|
| RAGAS faithfulness | 0.878 | 0.978 | +0.100 |
| RAGAS answer relevancy | 0.802 | 0.806 | +0.004 |
| Hit@1 | 0.767 | 0.900 | +0.133 |
| Hit@4 | 0.967 | 1.000 | +0.033 |
| MRR | 0.867 | 0.950 | +0.083 |
| Context precision (ID-based) | 0.850 | 0.914 | +0.064 |
| Context recall (ID-based) | 0.967 | 1.000 | +0.033 |
| Refusal accuracy | 0.975 | 1.000 | +0.025 |
| Correct refusals | 10/10 | 10/10 |  |
| False refusals | 1/30 | 0/30 |  |
| Answers with a citation | 0.966 | 1.000 | +0.034 |
| Answers citing the gold source | 0.897 | 1.000 | +0.103 |
| Invalid citations (total) | 0 | 0 | +0.000 |
| Retrieval ms/query | 49.196 | 255.552 | +206.357 |
