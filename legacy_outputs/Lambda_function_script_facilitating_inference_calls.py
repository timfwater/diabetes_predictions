import os
import io
import boto3
import json
import csv

# grab environment variables 
ENDPOINT_NAME = os.environ['ENDPOINT_NAME']
runtime= boto3.client('runtime.sagemaker')

def lambda_handler(event, context):
    print("Received event: " + json.dumps(event, indent=2))
    
    data = json.loads(json.dumps(event))
    payload = data['data']
    print(payload)
    
    response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                                       ContentType='text/csv',
                                       Body=payload)
    print(response)
    result = json.loads(response['Body'].read().decode())
    print(result)
    pred = int(result['predictions'][0]['predicted_label'])
    
    pred_rounded_by_cutoff = 1 if pred >=0.1027 else 0
    
    if(pred_rounded_by_cutoff == 0):
        return 'Not Readmitted'
    if(pred_rounded_by_cutoff == 1):
        return 'Readmitted'