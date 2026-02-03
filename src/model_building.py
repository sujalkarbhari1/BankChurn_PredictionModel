from config import Config
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

def model_building(df):
    models = {
    "Logistic Regression": LogisticRegression(),    
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "Bagging Classifier": BaggingClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "KNeighbors": KNeighborsClassifier(),
    "SCV": SVC()
}
    return models