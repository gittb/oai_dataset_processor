# OAI Dataset Processor

**OAI Dataset Processor** is a modular framework for processing large datasets using OpenAI-compatible endpoints. It provides SQL-based job persistence, worker-limited task distribution, and JSON schema validation.

## Installation

```bash
pip install oai-dataset-processor
```

## Key Features
- **Job Persistence**: Uses SQLite by default, configurable to any SQLAlchemy database
- **Bulk Processing**: Process multiple samples through OpenAI-compatible endpoints
- **Async Execution**: Semaphore-based worker limits for efficient job execution
- **JSON Schema Validation**: Enforce structured outputs using JSON schemas
- **Progress Monitoring**: Live progress bar for async tasks
- **Extensibility**: Easy to extend for custom storage or processing logic

## Quick Start

```python
from dataset_processor import OpenAIDatasetProcessor, create_runner_sample
from pydantic import BaseModel

# Define output schema
class SampleResponse(BaseModel):
    grade: int
    coherence: int

# Prepare samples
samples = [
    "The quick brown fox jumps over the lazy dog.",
    "What day today?",
    "The illusion of knowledge is the barrier to discovery.",
    "gpus go burrr"
]

job_samples = [
    create_runner_sample(
        job_id="job_123",
        model_name="gpt-4",
        instructions="Grade the sentence for grammar and coherence (1-10 each)",
        input_data=sample,
        output_json_schema=SampleResponse.model_json_schema(),
        sample_id=idx
    ) for idx, sample in enumerate(samples)
]

# Process samples
processor = OpenAIDatasetProcessor(
    base_url="YOUR_BASE_URL_HERE",
    api_key="YOUR_API_KEY_HERE",
    workers=20
)

processor.ingest_samples(job_samples)
results = processor.run_job("job_123")

# Export results
results.to_jsonl("output_results.jsonl")
print(processor.get_job_status("job_123"))
```

## Configuration

- **Database**: Default `sqlite:///datasetrunner.sqlite`. Configure via `db_url` in `OpenAIDatasetProcessor`
- **Parallelism**: Set concurrent workers via the `workers` parameter
- **Schema Validation**: Define output schemas using Pydantic models

## Dependencies
- `openai`
- `tqdm`
- `pandas`
- `sqlalchemy`
- `pydantic`

## Contributing
Contributions welcome! Please submit PRs for features, optimizations or documentation.