import json
import boto3

def split_s3_path(s3_path):
    path_parts=s3_path.replace("s3://","").split("/")
    bucket=path_parts.pop(0)
    key="/".join(path_parts)
    return bucket, key
    
def lambda_handler(event, context):
    pipeline_name = f"WebhookShowcase"
    body = json.loads(event['body'])
    url =  body['task']['data']['image']
    label = body['annotation']['result'][0]['value']['choices'][0]
    print(f'Label is {label} for image at {url}')
    s3 = boto3.resource("s3")
    s3_bucket = s3.Bucket('sagemaker-us-west-2-labelled')
    sbucket, skey = split_s3_path(url)
    copy_source = {
      'Bucket': sbucket,
      'Key': skey
    }
    bucket = s3.Bucket('sagemaker-us-west-2-*')
    bucket.copy(copy_source, f'annotated/{label}/{skey}')
    total = len([_.key for _ in s3_bucket.objects.all()])
    fire = total > 4 and total % 5 == 0
    
    name = 'a'
    if fire:
        client = boto3.client('sagemaker')
        execution = client.start_pipeline_execution(
                    PipelineName=pipeline_name)
        name = execution['PipelineExecutionArn']
        print('Pipeline has been executed :-)')
    else:
        print('Pipeline not executed...')
        
        
    return {
        'statusCode': 200,
        'isBase64Encoded': False,
        'body': {
            'msg': str(f'Fired {pipeline_name}' if fire else "Not fired")
        },
        'headers': {
            'Content-Type': 'application/json'
        }
    }
