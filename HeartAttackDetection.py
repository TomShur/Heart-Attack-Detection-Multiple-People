from keras.models import load_model


class HearAttackModel:

    def __init__(self, model_path):
        self.model_path = model_path
        self.model = load_model(self.model_path)

    def predict(self, skeleton):
        #Maybe do threading and predict only once every few frames or seconds
        if len(skeleton[0]) != 0:
            prediction = self.model.predict(skeleton)
            #Check what prediction is without the [0][0]
            return prediction[0][0]
        #return 0

    def confidence_level(self, prediction):
        if prediction != None:

            if prediction < 0 or prediction > 1:
                return "Something is very wrong"
            elif prediction >= 0 and prediction <= 0.5:
                return "No Heart attack"
            elif prediction <= 0.75:
                return "Potential Heart Attack"
            else:
                return "Heart Attack"
