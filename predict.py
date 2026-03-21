import numpy as np

def predict_dump(model_reg, model_clf, le, input_data):
    # Convert to array
    input_array = np.array(input_data).reshape(1, -1)

    # Regression prediction
    leftover = model_reg.predict(input_array)[0]

    # Classification prediction
    severity_encoded = model_clf.predict(input_array)[0]
    severity = le.inverse_transform([severity_encoded])[0]

    # Confidence (simple version)
    tree_preds = [tree.predict(input_array)[0] for tree in model_reg.estimators_]
    confidence = 100 - np.std(tree_preds)

    return leftover, severity, confidence