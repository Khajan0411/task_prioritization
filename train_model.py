# train_model.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
import joblib

def train_and_evaluate(X, y):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    cv_results = cross_validate(
        model, X, y, cv=5,
        scoring=['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'],
        return_estimator=True
    )
    
    print("Average Accuracy:", cv_results['test_accuracy'].mean())
    print("Average Precision:", cv_results['test_precision_macro'].mean())
    print("Average Recall:", cv_results['test_recall_macro'].mean())
    print("Average F1 Score:", cv_results['test_f1_macro'].mean())
    
    trained_model = model.fit(X, y)
    
    # Save the model to a file for deployment
    joblib.dump(trained_model, 'task_priority_model.pkl')
    
    return trained_model, cv_results['estimator'][0]
