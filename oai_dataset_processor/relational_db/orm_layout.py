from sqlalchemy import Column, String, JSON, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
import uuid

from oai_dataset_processor.models.sample_types import RunnerSample

Base = declarative_base()

class RunnerSampleDB(Base):
    __tablename__ = 'runner_samples'
    uid = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    id = Column(String, nullable=True)
    job_id = Column(String)
    model_name = Column(String)
    conversation = Column(JSON)
    target_format = Column(JSON)
    response = Column(JSON, nullable=True)
    processed = Column(Boolean, default=False)
    error = Column(Text, nullable=True)

    @classmethod
    def from_pydantic(cls, sample: RunnerSample):
        return cls(
            id=sample.id,
            job_id=sample.job_id,
            model_name=sample.model_name,
            conversation=sample.conversation,
            target_format=sample.target_format,
            response=sample.response,
            processed=sample.processed,
            error=sample.error
            )
    
    def to_pydantic(self) -> RunnerSample:
        return RunnerSample(
            id=self.id,
            job_id=self.job_id,
            model_name=self.model_name,
            conversation=self.conversation,
            target_format=self.target_format,
            response=self.response,
            processed=self.processed,
            error=self.error
        )

    def update_from_pydantic(self, sample: RunnerSample):
        self.id = sample.id
        self.job_id = sample.job_id
        self.model_name = sample.model_name
        self.conversation = sample.conversation
        self.target_format = sample.target_format
        self.response = sample.response
        self.processed = sample.processed
        self.error = sample.error
        return self