import json
import sagemaker
import base64
from sagemaker.serializers import IdentitySerializer
from sagemaker.predictor import Predictor 


 

# Fill this in with the name of your deployed model
ENDPOINT = 'image-classification-2024-12-06-22-04-21-827'## TODO: fill in

def lambda_handler(event, context):
   
    
    try:
        # Decode the image data
        image = base64.b64decode(event['body']['image_data'])
       
        
        # Instantiate a Predictor
        predictor = Predictor(endpoint_name=ENDPOINT,  
                              sagemaker_session=sagemaker.Session()) 
                              
     
        
        # For this model the IdentitySerializer needs to be "image/png"
        predictor.serializer = IdentitySerializer("image/png")
        
        # Make a prediction
        inference = predictor.predict(image)
         
        
        # Return the data back to the Step Function
        event["inferences"] = inference.decode('utf-8')
        
        return {
            'statusCode': 200,
            'body': json.dumps(event)
        }
        
    except Exception as e:
 
        return {
            'statusCode': 500,
            'body': json.dumps({"error": str(e)})
        }
