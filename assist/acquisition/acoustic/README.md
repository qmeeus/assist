# Acoustic model

The acoustic model is a model that converts a feature vector into some other
high level representation that is usually a distribution over components in
the model. The acoustic model is trained unsupervisadly on the audio only.
When using the model it takes a feature vector as input and returns the high
level representation as another vector. To create your own acoustic model
you can inherit from the general AcousticModel class defined in
acoustic_model.py and overwrite the abstract methods. Afterwards you should add
it to the factory method in acoustic_model_factory.py.
