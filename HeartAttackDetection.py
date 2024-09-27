from keras.models import load_model

#This class handles the prediction of the model we trained, and the final combined result


class HearAttackModel:
    def __init__(self, model_path):
        #Get model path and load it
        self.model_path = model_path
        self.model = load_model(self.model_path)

    def predict(self, skeleton):
        #Get the skeleton of an image and as long as it isn't empty predict the result and return them.
        #If the skeleton is empty returns None
        if len(skeleton[0]) != 0:
            prediction = self.model.predict(skeleton)
            #Check what prediction is without the [0][0]
            return prediction[0][0]
        #return 0

    def confidence_level(self, prediction):
        #Given a prediction (number), return the final diagnosis. Numbers are based on trial and error
        if prediction != None:

            if prediction < 0 or prediction > 1:
                return "Something is very wrong"
            elif prediction >= 0 and prediction <= 0.5:
                return "No Heart attack"
            elif prediction <= 0.75:
                return "Potential Heart Attack"
            else:
                return "Heart Attack"