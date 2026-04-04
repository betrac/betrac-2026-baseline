# Examples

Example output from the BeTraC omni baseline.

## Files

- [sample_output.jsonl](sample_output.jsonl) — Example JSONL output record
- [sample_output.txt](sample_output.txt) — Example SOAP note (the `summary` field content)

## Output Format

The pipeline produces JSONL with one record per audio sample:

```json
{
  "id": "sample_id",
  "summary": "S: ...\nO: ...\nA: ...\nP: ...",
  "thinking": "",
  "omni_time_sec": 92.2,
  "total_time_sec": 92.2,
  "success": true,
  "error": ""
}
```

The `summary` field contains a structured SOAP note. To customize, edit the prompt in [conf/prompt/default.yaml](../conf/prompt/default.yaml).
