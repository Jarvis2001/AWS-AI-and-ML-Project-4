#Lambda1:- Data Generation
import json
import boto3
import base64

s3 = boto3.resource("s3")

def lambda_handler(event, context):
    """A function to serialize target data from S3"""
    
    # Get the s3 address from the Step Function event input
    key =  "test/bicycle_s_000513.png"
    bucket = "sagemaker-us-east-1-673115746990"
    
    # Download the data from s3 to /tmp/image.png
    s3.Bucket(bucket).download_file(key, '/tmp/image.png')
    
    # We read the data from a file
    with open("/tmp/image.png", "rb") as f:
        image_data = base64.b64encode(f.read())

    # Pass the data back to the Step Function
    print("Event:", event.keys())
    return {
        'statusCode': 200,
        'body': {
            "image_data": image_data,
            "s3_bucket": bucket,
            "s3_key": key,
            "inferences": []
        }
    }

#Lambda2:- Classification
import json
import sagemaker
import base64
from sagemaker.serializers import IdentitySerializer

# Fill this in with the name of your deployed model
ENDPOINT = "image-classification-2023-08-10-20-31-16-413"  # TODO: Fill in your endpoint name

def lambda_handler(event, context):

    # Decode the image data
    image_data = event["body"]["image_data"]
    image = base64.b64decode(image_data)

    # Instantiate a Predictor
    predictor = sagemaker.predictor.Predictor(endpoint_name=ENDPOINT)

    # For this model the IdentitySerializer needs to be "image/png"
    predictor.serializer = IdentitySerializer("image/png")

    # Make a prediction
    inferences = predictor.predict(image)

    # Decode the binary data into a string
    inferences_str = inferences.decode("utf-8")

    # Parse the string into a list of floats
    inferences_list = json.loads(inferences_str)

    # We return the data back to the Step Function    
    event["body"]["inferences"] = inferences_list
    
    response_body = {
        "image_data": event["body"]['image_data'],
        "s3_bucket": event["body"]['s3_bucket'],
        "s3_key": event["body"]['s3_key'],
        "inferences": inferences_list
    }
    
    return {
        'statusCode': 200,
        'body': json.dumps(event)
    }


#Lambda3:- Inference Result
import json

THRESHOLD = 0.93

def lambda_handler(event, context):
    
    # Parse the JSON string and convert inferences to floats
    try:
        event_body = json.loads(event["body"])
        inferences = event_body["body"]["inferences"]
    except KeyError as e:
        return {
            'statusCode': 400,
            'body': json.dumps({'error': f'Missing key: {e}'})
        }
    
    # Check if any values in our inferences are above THRESHOLD
    meets_threshold = any(x >= THRESHOLD for x in inferences)
    
    # If our threshold is met, pass our data back out of the
    # Step Function, else, end the Step Function with an error
    if meets_threshold:
        response_body = {
            "image_data": event_body['body']['image_data'],
            "s3_bucket": event_body['body']['s3_bucket'],
            "s3_key": event_body['body']['s3_key'],
            "inferences": inferences
        }
        return {
            'statusCode': 200,
            'body': json.dumps(response_body)
        }
    else:
        return {
            'statusCode': 400,
            'body': json.dumps({'error': 'THRESHOLD_CONFIDENCE_NOT_MET'})
        }
