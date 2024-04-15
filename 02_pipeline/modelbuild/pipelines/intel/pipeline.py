
import os

import boto3
import sagemaker
import sagemaker.session

from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import (
    MetricsSource,
    ModelMetrics,
)
from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
    ScriptProcessor,
)
from sagemaker.pytorch.processing import PyTorchProcessor

from sagemaker.pytorch import PyTorchModel
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import (
    ConditionStep,
)
from sagemaker.workflow.functions import (
    JsonGet,
)
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
    ParameterFloat
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import (
    ProcessingStep,
    TrainingStep,
)
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.pipeline_context import PipelineSession

from sagemaker.pytorch import PyTorch
from sagemaker.debugger import TensorBoardOutputConfig
from sagemaker.workflow.steps import (
    ProcessingStep,
    TrainingStep,
)

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

####################
from datetime import datetime, timedelta
now = datetime.now() + timedelta(hours=5.50)
d = now.strftime("%d-%m-%Y-%H-%M-%S")
base_job_name = 'Intel'
job_name = f'{base_job_name}-{d}'
####################

def get_sagemaker_client(region):
    """Gets the sagemaker client.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        `sagemaker.session.Session instance
    """
    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")
    return sagemaker_client


def get_session(region, default_bucket):
    """Gets the sagemaker session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        `sagemaker.session.Session instance
    """

    boto_session = boto3.Session(region_name=region)

    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
        default_bucket=default_bucket,
    )

def get_pipeline_session(region, default_bucket):
    """Gets the pipeline session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        PipelineSession instance
    """

    boto_session = boto3.Session(region_name=region)
    sagemaker_client = boto_session.client("sagemaker")

    return PipelineSession(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        default_bucket=default_bucket,
    )

def get_pipeline_custom_tags(new_tags, region, sagemaker_project_arn=None):
    try:
        sm_client = get_sagemaker_client(region)
        response = sm_client.list_tags(
            ResourceArn=sagemaker_project_arn)
        project_tags = response["Tags"]
        for project_tag in project_tags:
            new_tags.append(project_tag)
    except Exception as e:
        print(f"Error getting project tags: {e}")
    return new_tags


def get_pipeline(
    region,
    sagemaker_project_arn=None,
    role=None,
    default_bucket=None,
    model_package_group_name="IntelPackageGroup",
    pipeline_name="IntelPipeline",
    base_job_prefix="Intel",
    processing_instance_type="ml.t3.medium",
    training_instance_type="ml.g4dn.2xlarge",
):
    """Gets a SageMaker ML Pipeline instance working with on Intel data.

    Args:
        region: AWS region to create and run the pipeline.
        role: IAM role to create and run steps and pipeline.
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        an instance of a pipeline
    """
    sagemaker_session = get_session(region, default_bucket)
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)

    pipeline_session = get_pipeline_session(region, default_bucket)
    
    # [START] intel pipeline
        
    input_dataset = ParameterString(
        name="InputDatasetZip",
        default_value="s3://sagemaker-us-west-2-***/intel.zip"
    )
    
    model_approval_status = ParameterString(
        name="ModelApprovalStatus",
        default_value="PendingManualApproval"
    )
    
    model_name = ParameterString(
        name="ModelName",
        default_value="resnet34"
    )
    
    batch_size = ParameterString(
        name="BatchSize",
        default_value="128"
    )
    
    opt_name = ParameterString(
        name="Optimizer",
        default_value="SGD"
    )
    
    lr = ParameterString(
        name="LearningRate",
        default_value="0.0018308341"
    )
    
    train_dataset = ParameterString(
        name="TrainDataset",
        default_value="Adam"
    )
    
    test_dataset = ParameterString(
        name="TestDataset",
        default_value="Adam"
    )
    
    base_job_name = base_job_prefix
    
    # PREPROCESS STEP
    
    pytorch_processor = PyTorchProcessor(
        framework_version='1.12',
        image_uri='***.dkr.ecr.us-west-2.amazonaws.com/emlo:infer',
        role=role,
        instance_type='ml.c5.xlarge',
        instance_count=1,
        sagemaker_session=pipeline_session,
        env={
            "GIT_USER": "m",
            "GIT_EMAIL": "m@emlo.com"
        }
    )
    
    processing_step_args = pytorch_processor.run(
        code='preprocess.py',
        source_dir=os.path.join(BASE_DIR, "scripts"),
        inputs=[
            ProcessingInput(
                input_name='data',
                source=input_dataset,
                destination='/opt/ml/processing/input'
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="train",
                source="/opt/ml/processing/dataset/train"
            ),
            ProcessingOutput(
                output_name="test",
                source="/opt/ml/processing/dataset/test"
            ),
        ],
        job_name=f"preprocess-{job_name}",
    )
    
    step_process = ProcessingStep(
        name="PreprocessIntelClassifierDataset",
        step_args=processing_step_args,
    )
    
    # TRAIN STEP
    
    tensorboard_output_config = TensorBoardOutputConfig(
        s3_output_path=f's3://{default_bucket}/sagemaker-intel-logs-pipeline',
        container_local_output_path='/opt/ml/output/tensorboard'
    )
    
    distribution = { 
        "pytorchddp": {
            "enabled": True,
            "custom_mpi_options": "-verbose -x NCCL_DEBUG=VERSION"
        }
    }
    
    
    pt_estimator = PyTorch(
        source_dir=os.path.join(BASE_DIR, "scripts"),
        entry_point="train.py",
        sagemaker_session=pipeline_session,
        role=role,
        image_uri='***.dkr.ecr.us-west-2.amazonaws.com/emlo:train',
        instance_count=1,
        instance_type="ml.g4dn.12xlarge",
        tensorboard_output_config=tensorboard_output_config,
        use_spot_instances=True,
        max_wait=1800,
        max_run=1500,
        disable_profiler=True, # for distributed training
        debugger_hook_config=False, # for distributed training
        distribution=distribution, # for DDP
        environment={
            "MODEL_NAME": model_name,
            "BATCH_SIZE": batch_size,
            "OPT_NAME": opt_name,
            "LR": lr,
            'USE_SMDEBUG':"0",
        }
    )
    
    
    estimator_step_args = pt_estimator.fit({
        'train': TrainingInput(
            s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                "train"
            ].S3Output.S3Uri,
        ),
        'test': TrainingInput(
            s3_data=step_process.properties.ProcessingOutputConfig.Outputs[
                "test"
            ].S3Output.S3Uri,
        )
        },
        job_name=f"training-{job_name}",
    )
    
    step_train = TrainingStep(
        name="TrainIntelClassifier",
        step_args=estimator_step_args,
    )
    
    # EVAL STEP
    
    pytorch_processor = PyTorchProcessor(
        image_uri='***.dkr.ecr.us-west-2.amazonaws.com/emlo:infer',
        role=role,
        sagemaker_session=pipeline_session,
        instance_type="ml.t3.xlarge",
        instance_count=1,
        framework_version='1.12',
    )

    eval_step_args = pytorch_processor.run(
        code='evaluate.py',
        source_dir=os.path.join(BASE_DIR, "scripts"),
        inputs=[
            ProcessingInput(
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=step_process.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
                destination="/opt/ml/processing/test",
            ),
        ],
        outputs=[
            ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation"),
            ProcessingOutput(output_name="data_drift", source="/opt/ml/processing/data_drift"),
            ProcessingOutput(output_name="explanation", source="/opt/ml/processing/explanation"),
            ProcessingOutput(output_name="robustness", source="/opt/ml/processing/robustness"),
        ],
        job_name=f"eval-{job_name}",
    )
    
    evaluation_report = PropertyFile(
        name="IntelClassifierEvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )
    step_eval = ProcessingStep(
        name="EvaluateIntelClassifierModel",
        step_args=eval_step_args,
        property_files=[evaluation_report],
    )

    # MODEL REGISTER STEP
    
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri="{}/evaluation.json".format(
                step_eval.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
            ),
            content_type="application/json"
        )
    )
    
    model = PyTorchModel(
        entry_point="infer.py",
        source_dir=os.path.join(BASE_DIR, "scripts"),
        sagemaker_session=pipeline_session,
        role=role,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        image_uri='***.dkr.ecr.us-west-2.amazonaws.com/emlo:infer',
        framework_version='1.12',
    )

    model_step_args = model.register(
        content_types=["application/json"],
        response_types=["application/json"],
        inference_instances=["ml.t2.medium"],
        transform_instances=["ml.m4.xlarge"],
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status,
        model_metrics=model_metrics,
    )
    
    step_register = ModelStep(
        name="RegisterIntelClassifierModel",
        step_args=model_step_args,
    )
    
    cond_gte = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step_name=step_eval.name,
            property_file=evaluation_report,
            json_path="multiclass_classification_metrics.accuracy.value"
        ),
        right=0.735,
    )

    step_cond = ConditionStep(
        name="CheckAccuracyIntelClassifierEvaluation",
        conditions=[cond_gte],
        if_steps=[step_register],
        else_steps=[],
    )


    # pipeline instance
    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[
            input_dataset,
            model_approval_status,
            model_name,
            batch_size,
            opt_name,
            lr,
        ],
        steps=[step_process, step_train, step_eval, step_cond],
        sagemaker_session=pipeline_session,
    )
    return pipeline
