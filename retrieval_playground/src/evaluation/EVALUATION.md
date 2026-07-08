# RAG Evaluation

## Layout

| File | Responsibility |
|------|----------------|
| `base.py` | Shared helpers (`keyword_overlap`, `mean_score`, …) |
| `ragas_runner.py` | RAGAS integration (`run_ragas`, `RAGEvaluator`) |
| `retrieval_metrics.py` | Hit@k, MRR, keyword overlap + optional RAGAS retrieval |
| `generation_metrics.py` | Faithfulness, answer relevancy, answer length ratio |
| `tool_metrics.py` | Tool / routing accuracy |
| `pipeline.py` | End-to-end orchestration |

## Notebook

`retrieval_playground/tutorial/4_Evaluation.ipynb`

## Notes

- **Classical retrieval metrics** compare chunks to `reference_context` (gold evidence).
- **RAGAS context_precision / context_recall** compare chunks to `reference` (gold **answer**). Do not pass `reference_context` as `reference`.
- **Generation metrics** use RAGAS with reference answers and generated responses.
