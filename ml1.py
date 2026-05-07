print(predicted[:20])
print(expected[:20])

wrong = [(p, e) for (p, e) in zip(predicted, expected) if p != e]

print(wrong)

print(f"Score: {knn.score(data_test, target_test)}")

from sklearn import confusion

confusion = confusion_matrix(y_true=expected, y_pred=predicted)

print(confusion)


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt2

confusion_df = pd.DataFrame(confusion, index=range(10), columns=range(10))

figure = plt2.figure(figsize=(7, 6))

axes = ssns.heatmap(confusion_df, annot=True, cmap=plt2.cm.nipy_spectral)
plt2.xlabel("expected")
plt2.ylabel("Predicted")
plt2.show()
