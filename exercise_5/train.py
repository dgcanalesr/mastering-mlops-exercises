import bentoml

from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


# Load data
data = load_iris(as_frame=True)
X = data["data"]
y = data["target"]

if __name__ == "__main__":

    # Fit model
    clf = QuadraticDiscriminantAnalysis()
    clf.fit(X,y)

    # Save model to BentoML
    saved_model = bentoml.sklearn.save_model(
        "iris_qda_clf", clf, signatures={"predict": {"batchable": True, "batch_dim": 0}}
        )
    
    print(f"Model saved: {saved_model}")
