from flask import Flask, render_template, request
import pandas as pd
import numpy as np
# Import FeatureEngineer class explicitly
from model import predict_single, get_input_features, num_cols, cat_cols, FeatureEngineer

app = Flask(__name__)

@app.route('/')
def home():
    tooltips = {
        'Age': 'Customer\'s age in years',
        'Tenure': 'Duration as customer in months',
        'Usage Frequency': 'Average monthly service usage',
        'Support Calls': 'Number of support calls made',
        'Payment Delay': 'Average payment delay in days',
        'Total Spend': 'Total amount spent by customer',
        'Last Interaction': 'Days since last customer interaction'
    }
    
    return render_template('form.html', num_cols=num_cols, tooltips=tooltips)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = {}
        
        # Process numerical features
        for feature in num_cols:
            try:
                value = float(request.form.get(feature))
                if value < 0:
                    return render_template('result.html',
                                         error=f"Negative value not allowed for {feature}",
                                         churn_status="Unknown",
                                         probability=0,
                                         recommendations=[])
                data[feature] = [value]
            except (TypeError, ValueError):
                return render_template('result.html',
                                     error=f"Invalid value for {feature}",
                                     churn_status="Unknown",
                                     probability=0,
                                     recommendations=[])
        
        # Process categorical features
        for feature in cat_cols:
            value = request.form.get(feature)
            if not value:
                return render_template('result.html',
                                     error=f"Missing value for {feature}",
                                     churn_status="Unknown",
                                     probability=0,
                                     recommendations=[])
            data[feature] = [value]
        
        # Convert to DataFrame
        input_df = pd.DataFrame(data)
        
        # Get prediction
        prediction, churn_prob = predict_single(input_df)
        probability = round(churn_prob * 100, 1)
        churn_status = "Likely to Churn" if prediction == 1 else "Not Likely to Churn"
        
        # Generate recommendations based on prediction
        if prediction == 1:
            recommendations = [
                "High Risk: Immediate intervention needed",
                "Schedule urgent customer review",
                "Prepare premium retention package",
                "Analyze usage patterns for pain points"
            ]
        else:
            recommendations = [
                "Low Risk: Maintain engagement",
                "Consider upgrade opportunities",
                "Schedule regular check-in",
                "Send appreciation message"
            ]
        
        return render_template('result.html',
                             churn_status=churn_status,
                             probability=probability,
                             recommendations=recommendations)
                             
    except Exception as e:
        return render_template('result.html',
                             error=f"Prediction error: {str(e)}",
                             churn_status="Unknown",
                             probability=0,
                             recommendations=[])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)