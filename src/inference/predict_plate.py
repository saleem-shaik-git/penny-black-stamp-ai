
"""
Inference entry point for Penny Black plate identification
Author: Saleem Shaik
"""

from model.plate_classifier import PlateClassifier

def predict_plate(features):
    classifier = PlateClassifier()
    return classifier.predict(features)
