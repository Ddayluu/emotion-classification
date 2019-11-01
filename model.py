from keras.models import model_from_json
import numpy as np

class FacialExpressionModel(object):
	EMOTION_LIST = ["Angry", "Disgust", "Fear",
					"Happy", "Neutral", "Sad", "Surprise"]

	def __init__(self, json_model, weights_model):
		with open(json_model, "r") as json_file:
			loaded_model_json = json_file.read()
			self.loaded_model = model_from_json(loaded_model_json)

		# Load weights 
		self.loaded_model.load_weights(weights_model)
		self.loaded_model._make_predict_function()


	def predict_emotion(self, img):
		self.preds = self.loaded_model.predict(img)
		return FacialExpressionModel.EMOTION_LIST[np.argmax(self.preds)]