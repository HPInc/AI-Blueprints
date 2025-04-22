from pathlib import Path
import pandas as pd
import numpy as np  

# Generate comprehensive banking dataset and save it as a single CSV
# Create data directory
data_dir = Path('data')
data_dir.mkdir(exist_ok=True)

def generate_comprehensive_banking_dataset(num_customers=1000):
    print(f"Generating comprehensive banking dataset for {num_customers} customers...")
    
    # Product categories
    products = [
        'premium_credit_card', 
        'mortgage_refinance', 
        'investment_account',
        'high_yield_savings', 
        'personal_loan'
    ]
    
    # Customer segments
    segments = [
        'Premium', 
        'Standard', 
        'Basic', 
        'Student', 
        'Senior'
    ]
    
    # Risk profiles
    risk_profiles = [
        'Conservative',
        'Moderate',
        'Aggressive',
        'Very Conservative',
        'Very Aggressive'
    ]
    
    # Generate the main customer dataset
    dataset = pd.DataFrame({
        'customer_id': range(1, num_customers + 1),
        'age': np.random.randint(18, 80, num_customers),
        'income': np.round(np.random.normal(70000, 30000, num_customers), 2),
        'credit_score': np.random.randint(300, 850, num_customers),
        'account_balance': np.round(np.random.lognormal(10, 1, num_customers), 2),
        'tenure_months': np.random.randint(1, 240, num_customers),
        'num_products': np.random.randint(1, 5, num_customers),
        'active_investment': np.random.randint(0, 2, num_customers),
        'has_mortgage': np.random.randint(0, 2, num_customers),
        'has_loan': np.random.randint(0, 2, num_customers),
        'has_credit_card': np.random.randint(0, 2, num_customers),
        'segment': np.random.choice(segments, num_customers),
        'risk_profile': np.random.choice(risk_profiles, num_customers),
        'total_assets': np.round(np.random.lognormal(11, 1.5, num_customers), 2),
        'monthly_expenses': np.round(np.random.normal(3000, 1500, num_customers), 2),
        'last_interaction_days': np.random.randint(1, 100, num_customers)
    })
    
    # Add product ownership and potential recommendation flags
    for product in products:
        # Current ownership
        dataset[f'has_{product}'] = np.random.randint(0, 2, num_customers)
        
        # Recommendation strength (0-100 score)
        dataset[f'recommend_{product}'] = np.random.randint(0, 101, num_customers)
        
        # For customers who already have the product, set recommendation to 0
        dataset.loc[dataset[f'has_{product}'] == 1, f'recommend_{product}'] = 0
    
    # Calculate monthly income and round to 2 decimal places
    dataset['monthly_income'] = np.round(dataset['income'] / 12, 2)
    
    # Calculate debt-to-income ratio
    dataset['debt_to_income'] = np.round(np.random.uniform(0.1, 0.6, num_customers), 2)
    
    # Calculate savings ratio
    dataset['savings_ratio'] = np.round((dataset['monthly_income'] - dataset['monthly_expenses']) / dataset['monthly_income'], 2)
    dataset.loc[dataset['savings_ratio'] < 0, 'savings_ratio'] = 0
    dataset.loc[dataset['savings_ratio'] > 0.7, 'savings_ratio'] = 0.7
    
    # Create text descriptions for each customer
    descriptions = []
    for _, row in dataset.iterrows():
        desc = f"""Customer Profile:
Age: {row['age']}
Income: ${row['income']:.2f} yearly (${row['monthly_income']:.2f} monthly)
Credit Score: {row['credit_score']}
Account Balance: ${row['account_balance']:.2f}
Total Assets: ${row['total_assets']:.2f}
Monthly Expenses: ${row['monthly_expenses']:.2f}
Savings Ratio: {row['savings_ratio']:.2f}
Debt-to-Income Ratio: {row['debt_to_income']:.2f}
Tenure: {row['tenure_months']} months
Current Products: {row['num_products']}
Customer Segment: {row['segment']}
Risk Profile: {row['risk_profile']}
Last Interaction: {row['last_interaction_days']} days ago
Active Investment: {'Yes' if row['active_investment'] else 'No'}
Has Mortgage: {'Yes' if row['has_mortgage'] else 'No'}
Has Loan: {'Yes' if row['has_loan'] else 'No'}
Has Credit Card: {'Yes' if row['has_credit_card'] else 'No'}
Premium Credit Card: {'Yes' if row['has_premium_credit_card'] else 'No'}
Mortgage Refinance: {'Yes' if row['has_mortgage_refinance'] else 'No'}
Investment Account: {'Yes' if row['has_investment_account'] else 'No'}
High Yield Savings: {'Yes' if row['has_high_yield_savings'] else 'No'}
Personal Loan: {'Yes' if row['has_personal_loan'] else 'No'}
"""
        descriptions.append(desc)
    
    # Add the descriptions to the dataset
    dataset['customer_description'] = descriptions
    
    # Save the comprehensive dataset
    dataset.to_csv(data_dir / 'banking_dataset.csv', index=False)
    
    print(f"Comprehensive banking dataset saved to {data_dir / 'banking_dataset.csv'}")
    
    return dataset

# Generate the comprehensive dataset
banking_dataset = generate_comprehensive_banking_dataset(1000)