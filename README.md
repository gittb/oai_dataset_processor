# OAI Dataset Processor

**OAI Dataset Processor** is a modular, scalable, and fault-tolerant framework designed for processing large datasets or tasks using OpenAI-compatible endpoints. It provides robust support for job persistence through SQL databases, effective task distribution with worker limits, and smooth integration of JSON schema-based output validation.

---

## Key Features
- **Job Persistence**: Uses SQLite by default (configurable) to ensure job data survives crashes or errors.
- **Bulk Processing**: Supports ingestion, storage, and processing of multiple samples using an OpenAI-compatible endpoint.
- **Async Execution**: Uses Python’s `asyncio` with semaphore-based worker limits for efficient job execution.
- **JSON Schema Validation**: Enforces structured outputs using flexible JSON schema definitions.
- **Progress Monitoring**: Displays async task progress with a live progress bar.
- **Extensibility**: Easy to expand for custom storage backends (e.g., Postgres) or additional processing logic.

---

## Installation

To install the package, simply clone the repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

---

## Usage

### Code Example
```python
from dataset_processor import OpenAIDatasetProcessor, create_runner_sample
from pydantic import BaseModel

# Configure instructions and JSON schema
sample_instructions = "Please grade the sentence for grammar and coherence, 1-10 for each, respond with JSON."

class SampleResponse(BaseModel):
    grade: int
    coherence: int

json_schema = SampleResponse.model_json_schema()

# Prepare input samples
input_samples = [
    "The quick brown fox jumps over the lazy dog.",
    "What day today?",
    "The illusion of knowledge is the barrier to discovery.",
    "gpus go burrr"
]

samples = []
for idx, input_sample in enumerate(input_samples):
    samples.append(create_runner_sample(
        job_id="job_123",
        model_name="gpt-4",
        instructions=sample_instructions,
        input_data=input_sample,
        output_json_schema=json_schema,
        sample_id=idx
    ))

# Create the processor and ingest samples
processor = OpenAIDatasetProcessor(
    base_url="http://api.openai.compatible.endpoint/api",
    api_key="YOUR_API_KEY",
    workers=20
)
processor.ingest_samples(samples)

# Run the job and retrieve results
results = processor.run_job("job_123")

# Save results to JSONL
results.to_jsonl("output_results.jsonl")

# View Job Status
print(processor.get_job_status("job_123"))
```

---

## Configurations

- **Job Persistence**: By default, jobs are stored in `sqlite:///datasetrunner.sqlite`. To use a different database:
  - Pass the desired `db_url` to `OpenAIDatasetProcessor` or `StorageHandler`.
  - Supported DBs: SQLite, PostgreSQL, and other SQLAlchemy-compatible backends.

- **Async Execution and Parallelism**: Configure the number of workers using the `workers` parameter. The worker limit is enforced via Python’s `asyncio.Semaphore`.

- **Custom Output Validation**: Define reusable validation schemas using Pydantic models. For instance, the `SampleResponse` model provides a consistent structure for score-based outputs.

---

## Key Classes and Functions:

- **`OpenAIDatasetProcessor`**: Main class to handle ingestion, processing, and retrieval of jobs and samples.
- **`StorageHandler`**: Handles database operations for retrieving, saving, and managing datasets.
- **`create_runner_sample`**: Simplifies the creation of samples with job-specific metadata.

---

## Dependencies

The following dependencies are required (managed via `requirements.txt`):
- `openai`
- `tqdm`
- `pandas`
- `sqlalchemy`
- `pydantic`

---

## Contributing

Contributions are welcome! Whether you want to add new features, optimize performance, or expand documentation, feel free to fork the repository and submit a pull request.

--- 

