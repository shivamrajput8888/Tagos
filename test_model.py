from predict import predict_tags

sample = """
Python is widely used in Machine Learning and Artificial Intelligence.
Data Science uses Python for data analysis.
"""

results = predict_tags(sample)

print()

print("Predicted Tags")

print()

for tag, score in results:

    print(f"{tag} : {score}%")