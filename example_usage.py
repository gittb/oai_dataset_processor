from oai_dataset_processor import OpenAIDatasetProcessor, create_runner_sample
from pydantic import BaseModel

# Configuring the sample data, some instructions and a target schema to fill

sample_instructions = "Please grade the sentence for grammar and coherence, 1-10 for each, respond with json"

class SampleResponse(BaseModel):
    grade: int
    coherence: int

json_schema = SampleResponse.model_json_schema()

input_samples = [
    'The quick brown fox jumps over the lazy dog.',
    'What day today',
    'The illusion of knowldge is the barrier to discovery',
    'gpus go burrr'
]

# Creating the samples via the create_runner_sample function

samples = []

for idx, samp in enumerate(input_samples):
    sample = create_runner_sample(
        job_id="test_job",
        model_name="YOUR_MODELS_NAME_HERE",
        instructions=sample_instructions,
        input_data=samp,
        output_json_schema=json_schema,
        sample_id=idx
    )
    samples.append(sample)

runner = OpenAIDatasetProcessor(
    base_url="YOUR_BASE_URL_HERE",
    api_key="YOUR_API_KEY_HERE",
    workers=20
)

runner.ingest_samples(samples) # This intakes the samples into the DB

# we can grab the statuus of the job, including number of samples processed and unprocessed
print(runner.get_job_status("test_job"))
# >>> {'test_job': {'processed': 0, 'unprocessed': 4}}

# we can also retrieve the samples from the DB and convert them to a dataframe for analysis
print(runner.get_job_samples("test_job").to_dataframe().head())

# we can start running the job, which will process the samples and return the results
results = runner.run_job("test_job")

# this results object (JobResult) is the same as get_job_sample and can be sent to df or jsonl
print(results.to_dataframe().head())
results.to_jsonl("DatasetProcessorDB/test_results.jsonl")

# Finally reprint the job status to see the final status of the job. It should be completed.
print(runner.get_job_status("test_job"))
# >>> {'test_job': {'processed': 4, 'unprocessed': 0}}

