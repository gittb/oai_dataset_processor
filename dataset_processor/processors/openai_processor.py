
from openai import AsyncOpenAI
from typing import List, Dict
from tqdm.asyncio import tqdm_asyncio

from dataset_processor.models.sample_types import RunnerSample, JobResult

from dataset_processor.relational_db.storage_handler import StorageHandler
from dataset_processor.relational_db.orm_layout import RunnerSampleDB

import asyncio
import json


class OpenAIDatasetProcessor:
    def __init__(self, base_url, api_key, workers:int, db_url: str=None):
        self.create_async_client(base_url, api_key)
        self.storage_handler = StorageHandler(db_url=db_url)

        self.workers = workers

    def create_async_client(self, base_url, api_key):
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key
        )

    def ingest_samples(self, samples: List[RunnerSample]):
        """
        Ingests a list of RunnerSample instances into the database.
        Parameters:
            samples (List[RunnerSample]): A list of RunnerSample instances to be ingested.
        """
        self.storage_handler.add_bulk_pydantic_samples(samples)

    def get_job_samples(self, job_id):
        """
        Retrieves all samples associated with a specific job ID from the database.
        Parameters:
            job_id (str): The ID of the job for which to retrieve samples.
        Returns:
            List[RunnerSample]: A list of RunnerSample instances associated with the job ID.
        """
        return self.storage_handler.get_job_samples(job_id, only_unprocessed=False)
    
    def get_job_status(self, job_id):
        """
        Retrieves the status of a specific job ID from the database.
        Parameters:
            job_id (str): The ID of the job for which to retrieve the status.
        Returns:
            dict: A dictionary containing the status of the job.
        """
        return self.storage_handler.get_job_status(job_id)
    
    def get_jobs(self):
        """
        Retrieves all jobs from the database.
        Returns:
            List[dict]: A list of dictionaries containing job information.
        """
        return self.storage_handler.get_jobs()

    def reset_errors(self, job_id: str):
        """
        Resets the error status of all samples associated with a specific job ID.
        Parameters:
            job_id (str): The ID of the job for which to reset the error status.
        """
        self.storage_handler.reset_errors(job_id)

    async def run_sample(self, sample:RunnerSample):
        try:
            chat_completion = await self.client.chat.completions.create(
                messages=sample.conversation,
                temperature=0,
                model=sample.model_name,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "target", "schema": sample.target_format},
                },
            )
            message = chat_completion.choices[0].message.content
            
            # create parsed model here
            sample.response = json.loads(message) # could probably speed up with orjson
            sample.processed = True
            sample.error = None
        except Exception as e:
            sample.response = None
            sample.processed = True # flag as processed to avoid reprocessing in case of failure
            sample.error = str(e)  # Store the error message for debugging

        return sample

    def run_job(self, job_id: str) -> JobResult:
        """
        Processes all unprocessed samples associated with a specific job ID. 

        Parameters:
            job_id (str): The ID of the job to process. Used to fetch the relevant samples from the database.

        Returns:
            JobResult : A list of processed samples with their responses.
        """
        async def run_job_async():
            semaphore = asyncio.Semaphore(self.workers)

            async def process_sample(sample: RunnerSample):
                """Helper coroutine to process a single sample."""
                async with semaphore:
                    await self.run_sample(sample)  # Process the sample (updates `response` field internally)
                    self.storage_handler.save_sample_to_db(RunnerSampleDB.from_pydantic(sample))
                    return sample

            # Fetch samples from the database
            job_samples = self.storage_handler.get_job_samples(job_id, only_unprocessed=True)

            # Use tqdm.asyncio to wrap the tasks with a progress bar
            tasks = [process_sample(sample) for sample in job_samples.samples]

            # Wait for tasks to complete with a progress bar
            await tqdm_asyncio.gather(*tasks, desc="Processing Samples", total=len(job_samples.samples))

            print(f"Processed {len(job_samples.samples)} samples for job {job_id}.")

            # Return the list of processed samples
            return JobResult(job_id=job_id, samples=job_samples.samples)

        # Run the async logic through asyncio.run()
        return asyncio.run(run_job_async())