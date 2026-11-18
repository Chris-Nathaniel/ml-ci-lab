import numpy as np
from sklearn.metrics import accuracy_score
from train import load_data, train

MIN_ACCURACY = 0.95

def test_data_shape():
  X_train, X_test, _, _ = load_data()
  assert not X_train.isna().any().any()
  assert not X_test.isna().any().any()

def test_prediction_shape():
  X_train, X_test, y_train, _ = load_data()
  preds = train(X_train, y_train).predict(X_test)
  assert preds.shape == (len(X_test),)
  assert set(np.unique(preds)) <= {0, 1}

def test_training_is_reproducible():
  X_train, X_test, y_train,_ = load_data()
  model1 = train(X_train, y_train).predict(X_test)
  model2 = train(X_train, y_train).predict(X_test)
  assert np.array_equal(model1, model2)

def test_accuracy_above_threshold():
  X_train, X_test, y_train, y_test = load_data()
  model = train(X_train, y_train)
  preds = model.predict(X_test)
  accuracy = accuracy_score(y_test, preds)
  assert accuracy >= MIN_ACCURACY, f"accuracy{accuracy:.4f} is below {MIN_ACCURACY}" 
