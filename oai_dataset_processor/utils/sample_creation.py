from oai_dataset_processor.models.sample_types import RunnerSample

def create_runner_sample(
        job_id:str,
        model_name:str,
        instructions:str,
        input_data:str,
        output_json_schema:str,
        sample_id:str=None,
) -> RunnerSample:
    """
    Creates a RunnerSample object with the provided parameters.
    Args:
        job_id (str): The ID of the job. All samples in the job will have the same job_id.
        model_name (str): The name of the model.
        instructions (str): The instructions for the model.
        input_data (str): The input data for the model.
        output_json_schema (str): The JSON schema for the model's output.
        sample_id (str, optional): The ID of the sample. Defaults to None.
    Returns:
        RunnerSample: A RunnerSample object with the provided parameters.
    """
    convo = [
        {
            "role": "system",
            "content": instructions,
        },
        {
            "role": "user",
            "content": input_data,
        },
    ]

    samp = RunnerSample(
        job_id=job_id,
        model_name=model_name,
        conversation=convo,
        target_format=output_json_schema,
    )
    if sample_id:
        samp.id = sample_id
    return samp