from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .orm_layout import Base, RunnerSampleDB
from oai_dataset_processor.models.sample_types import RunnerSample, JobResult
from typing import Dict, List, Union
import os

class StorageHandler:
    def __init__(self, db_url=None):
        """
        Initializes the StorageHandler with a database URL.
        If no URL is provided, it defaults to a SQLite database in the current working directory.
        """
        if db_url is None:
            base_dir = os.path.join(os.getcwd(), "DatasetProcessorDB")
            os.makedirs(base_dir, exist_ok=True)
            db_url = f"sqlite:///{os.path.join(base_dir, 'datasetrunner.sqlite')}"

        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def save_sample_to_db(self, sample: RunnerSample):
        """
        Saves a RunnerSample object to the database.
        """
        with self.Session() as session:
            existing = session.query(RunnerSampleDB).filter_by(id=sample.id).first()
            if not existing:
                db_sample = RunnerSampleDB.from_pydantic(sample)
                session.add(db_sample)
                session.commit()
            else:
                existing.update_from_pydantic(sample)
                session.commit()

    def get_job_samples(self, job_id, only_unprocessed=False) -> JobResult:
        """
        Returns a list of job samples for a given job_id.
        """
        with self.Session() as session:
            query = session.query(RunnerSampleDB).filter_by(job_id=job_id)
            if only_unprocessed:
                query = query.filter_by(processed=False)
            samples = query.all()
            pydantic_samples = [sample.to_pydantic() for sample in samples]
            return JobResult(job_id=job_id, samples=pydantic_samples)

    def add_bulk_pydantic_samples(self, samples: RunnerSample):
        """
        Adds multiple RunnerSample objects to the database.
        """
        for sample in samples:
            db_sample = RunnerSampleDB.from_pydantic(sample)
            self.save_sample_to_db(db_sample)

    def get_job_status(self, job_id) -> Dict[str, Dict[str, int]]:
        """
        Returns a single job_id and their status (number of processed and unprocessed samples).
        """
        with self.Session() as session:
            query = session.query(RunnerSampleDB).filter_by(job_id=job_id)
            results = query.all()
            status = {}
            for sample in results:
                if sample.job_id not in status:
                    status[sample.job_id] = {'processed': 0, 'unprocessed': 0}
                if sample.processed:
                    status[sample.job_id]['processed'] += 1
                else:
                    status[sample.job_id]['unprocessed'] += 1
            return status
        
    def get_jobs(self) -> List[Dict[str, int]]:
        """
        Returns a list of all unique job IDs in the database and sample counts
        """
        jobs = {}
        with self.Session() as session:
            query = session.query(RunnerSampleDB.job_id).distinct()
            job_ids = [job_id[0] for job_id in query.all()]
            # Count samples for each job_id
            for job_id in job_ids:
                count_query = session.query(RunnerSampleDB).filter_by(job_id=job_id).count()
                jobs[job_id] = int(count_query)
        return jobs

    def delete_job(self, job_id):
        """
        Deletes all samples associated with a given job_id.
        """
        with self.Session() as session:
            session.query(RunnerSampleDB).filter_by(job_id=job_id).delete()
            session.commit()

    def get_sample_by_id(self, sample_id) -> Union[RunnerSample, None]:
        """
        Retrieves a sample by its ID.
        """
        with self.Session() as session:
            sample = session.query(RunnerSampleDB).filter_by(id=sample_id).first()
            if sample:
                return sample.to_pydantic()
            return None
        
    def reset_errors(self, job_id):
        """
        Resets the processed flag for all samples in a job that have an error.
        """
        with self.Session() as session:
            session.query(RunnerSampleDB).filter_by(job_id=job_id, error=True).update({'processed': False, 'error': None})
            session.commit()