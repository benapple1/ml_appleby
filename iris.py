# The Iris dataset is referred to as a “toy dataset” because it has only 150 samples and four features.
# The dataset describes 50 samples for each of three Iris flower species—Iris setosa, Iris versicolor and Iris
# virginica. Each sample’s features are the sepal length, sepal width, petal
# length and petal width, all measured in centimeters. The sepals are the larger outer parts of each flower
# that protect the smaller inside petals before the flower buds bloom.

# EXERCISE
# load the iris dataset and use classification
# to see if the expected and predicted species
# match up

# display the shape of the data, target and target_names

# display the first 10 predicted and expected results using
# the species names not the number (using target_names)

# display the values that the model got wrong

# visualize the data using the confusion matrix


from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load the dataset
iris = load_iris()

# Display the shape of data, target, and target_names
print("Data shape:", iris.data.shape)
print("Target shape:", iris.target.shape)
print("Target names:", iris.target_names)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, random_state=11
)

# Train KNN classifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

# Generate predictions
predicted = knn.predict(X_test)
expected = y_test

# Display first 10 predicted and expected using species names
print("\nFirst 10 Predictions vs Expected:")
for i in range(10):
    pred_name = iris.target_names[predicted[i]]
    exp_name = iris.target_names[expected[i]]
    print(f"Predicted: {pred_name}, Expected: {exp_name}")

# Display incorrect value
print("\nMisclassified samples:")
for i in range(len(predicted)):
    if predicted[i] != expected[i]:
        pred_name = iris.target_names[predicted[i]]
        exp_name = iris.target_names[expected[i]]
        print(f"Sample {i} — Predicted: {pred_name}, Expected: {exp_name}")

# Confusion matrix
cm = confusion_matrix(expected, predicted)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot()
plt.xlabel("Expected")
plt.ylabel("Predicted")
plt.show()
