import pandas as pd
import numpy as np
from faker import Faker
import random
from pathlib import Path
import os

# Initialize Faker
fake = Faker()
Faker.seed(42)  # For reproducibility
np.random.seed(42)
random.seed(42)

# Define constants
NUM_USERS = 1000
OUTPUT_FILE = "data/tech_adoption_dataset.csv"
# EMBEDDINGS_FILE = "data/tech_embeddings.npy"  # This doesn't belong here

# Create data directory if it doesn't exist
os.makedirs("data", exist_ok=True)

# Define technology categories and specific products
TECH_CATEGORIES = {
    "Smartphones": ["iPhone", "Samsung Galaxy", "Google Pixel", "OnePlus", "Xiaomi"],
    "Laptops": ["MacBook", "Dell XPS", "Lenovo ThinkPad", "HP Spectre", "ASUS ZenBook"],
    "Smart Home": ["Amazon Echo", "Google Nest", "Apple HomePod", "Samsung SmartThings", "Philips Hue"],
    "Wearables": ["Apple Watch", "Fitbit", "Samsung Galaxy Watch", "Garmin", "Oura Ring"],
    "VR/AR": ["Meta Quest", "HTC Vive", "PlayStation VR", "Microsoft HoloLens", "Apple Vision Pro"]
}

# Adoption categories
ADOPTER_CATEGORIES = ["Innovator", "Early Adopter", "Early Majority", "Late Majority", "Laggard"]
ADOPTER_WEIGHTS = [0.025, 0.135, 0.34, 0.34, 0.16]  # Based on Rogers' diffusion of innovations theory

# Technology interest levels
TECH_INTEREST_LEVELS = ["Very High", "High", "Moderate", "Low", "Very Low"]
TECH_INTEREST_WEIGHTS = [0.1, 0.25, 0.4, 0.15, 0.1]

# Budget categories
BUDGET_CATEGORIES = ["Premium", "High", "Mid-range", "Budget", "Economy"]
BUDGET_WEIGHTS = [0.1, 0.2, 0.4, 0.2, 0.1]

# Digital platforms
DIGITAL_PLATFORMS = ["Instagram", "YouTube", "Twitter", "LinkedIn", "TikTok", "Reddit", "Facebook", "Email Newsletter", "Tech Blogs"]

# Purchasing factors
PURCHASING_FACTORS = ["Price", "Features", "Brand Reputation", "Design", "Reviews", "Sustainability", "Compatibility"]

def generate_user_id(index):
    """Generate a unique user ID"""
    return f"TU{index+1:04d}"

def create_user_description(user_data):
    """
    Create a comprehensive text description of a user for embedding
    This combines all relevant user data into a single string
    """
    # Personal demographics
    demographics = f"User ID: {user_data['user_id']}, Age: {user_data['age']}, Gender: {user_data['gender']}, "
    demographics += f"Location: {user_data['location']}, Education: {user_data['education']}, "
    demographics += f"Job Sector: {user_data['job_sector']}, Annual Income: ${user_data['annual_income']:.2f}, "
    
    # Technology profile
    tech_profile = f"Adopter Category: {user_data['adopter_category']}, "
    tech_profile += f"Tech Interest: {user_data['tech_interest']}, Budget Category: {user_data['budget_category']}, "
    tech_profile += f"Technical Proficiency: {user_data['technical_proficiency']}, "
    
    # Purchasing behavior
    purchasing = f"Primary Purchasing Factor: {user_data['primary_purchasing_factor']}, "
    if pd.notna(user_data['secondary_purchasing_factor']):
        purchasing += f"Secondary Purchasing Factor: {user_data['secondary_purchasing_factor']}, "
    purchasing += f"Preferred Platforms: {user_data['preferred_platforms']}, "
    
    # Technology adoption patterns
    adoption = "Technology Adoption: "
    
    # Add smartphone adoption info
    if user_data['smartphone_adoption_timing'] >= 0:
        adoption += f"Adopts smartphones within {user_data['smartphone_adoption_timing']} months "
        adoption += f"(Prefers {user_data['smartphone_preferred_product']}), "
    else:
        adoption += "Does not use smartphones, "
        
    # Add laptop adoption info
    if user_data['laptop_adoption_timing'] >= 0:
        adoption += f"Adopts laptops within {user_data['laptop_adoption_timing']} months "
        adoption += f"(Prefers {user_data['laptop_preferred_product']}), "
    else:
        adoption += "Does not use laptops, "
        
    # Add smart home adoption info
    if user_data['smarthome_adoption_timing'] >= 0:
        adoption += f"Adopts smart home tech within {user_data['smarthome_adoption_timing']} months "
        adoption += f"(Prefers {user_data['smarthome_preferred_product']}), "
    else:
        adoption += "Does not use smart home technology, "
        
    # Add wearable adoption info
    if user_data['wearable_adoption_timing'] >= 0:
        adoption += f"Adopts wearables within {user_data['wearable_adoption_timing']} months "
        adoption += f"(Prefers {user_data['wearable_preferred_product']}), "
    else:
        adoption += "Does not use wearables, "
        
    # Add VR/AR adoption info
    if user_data['vrar_adoption_timing'] >= 0:
        adoption += f"Adopts VR/AR within {user_data['vrar_adoption_timing']} months "
        adoption += f"(Prefers {user_data['vrar_preferred_product']})"
    else:
        adoption += "Does not use VR/AR technology"
    
    # Combine all sections
    full_description = demographics + tech_profile + purchasing + adoption
    return full_description

def generate_tech_adoption_profile(user_id):
    """Generate a technology adoption profile for one user"""
    age = random.randint(18, 75)
    
    # Weight adopter category based on age
    if age < 30:
        # Younger users more likely to be earlier adopters
        adopter_weights = [0.15, 0.35, 0.30, 0.15, 0.05]
    elif age > 55:
        # Older users more likely to be later adopters
        adopter_weights = [0.05, 0.10, 0.25, 0.40, 0.20]
    else:
        # Use standard weights for middle-aged users
        adopter_weights = ADOPTER_WEIGHTS
        
    adopter_category = random.choices(ADOPTER_CATEGORIES, weights=adopter_weights, k=1)[0]
    
    # Tech interest level correlates with adopter category
    tech_interest_map = {
        "Innovator": ["Very High", "High"],
        "Early Adopter": ["Very High", "High", "Moderate"],
        "Early Majority": ["High", "Moderate"],
        "Late Majority": ["Moderate", "Low"],
        "Laggard": ["Low", "Very Low"]
    }
    tech_interest = random.choice(tech_interest_map[adopter_category])
    
    # Budget correlates with age (bell curve) and adopter category
    income_factor = min(1.0, (age - 18) / 30)  # Peaks around age 48
    income_factor = income_factor * (2 - income_factor)  # Create bell curve
    
    # Adjust income based on adopter category
    adopter_budget_boost = {
        "Innovator": 0.8,
        "Early Adopter": 0.4,
        "Early Majority": 0.0,
        "Late Majority": -0.2,
        "Laggard": -0.4
    }
    
    income_factor = max(0.1, min(1.0, income_factor + adopter_budget_boost[adopter_category]))
    # Convert to float with 2 decimal places for cents
    annual_income = round(35000 + (income_factor * 165000) + random.random() * 100, 2)
    
    # Determine budget category based on income and adopter profile
    if "Innovator" in adopter_category or "Early Adopter" in adopter_category:
        budget_weights = [0.3, 0.4, 0.2, 0.08, 0.02]
    elif "Early Majority" in adopter_category:
        budget_weights = [0.05, 0.25, 0.5, 0.15, 0.05]
    elif "Late Majority" in adopter_category:
        budget_weights = [0.02, 0.08, 0.4, 0.4, 0.1]
    else:  # Laggard
        budget_weights = [0.01, 0.04, 0.15, 0.4, 0.4]
        
    budget_category = random.choices(BUDGET_CATEGORIES, weights=budget_weights, k=1)[0]
    
    # Preferred platforms based on age and adopter category
    if age < 25:
        platform_choices = ["TikTok", "Instagram", "YouTube", "Twitter", "Reddit"]
    elif age < 40:
        platform_choices = ["Instagram", "YouTube", "Twitter", "LinkedIn", "Reddit", "TikTok"]
    elif age < 55:
        platform_choices = ["LinkedIn", "Facebook", "YouTube", "Email Newsletter", "Twitter", "Tech Blogs"]
    else:
        platform_choices = ["Facebook", "Email Newsletter", "YouTube", "Tech Blogs"]
        
    # Early adopters more likely to use more diverse platforms
    num_platforms = max(1, min(5, int(6 - ADOPTER_CATEGORIES.index(adopter_category))))
    preferred_platforms = random.sample(platform_choices, k=min(num_platforms, len(platform_choices)))
    
    # Generate adoption timing for each tech category
    adoption_timing = {}
    adoption_products = {}
    
    for category, products in TECH_CATEGORIES.items():
        # Determine if user adopts this category
        adoption_probability = {
            "Innovator": 0.95,
            "Early Adopter": 0.85,
            "Early Majority": 0.7,
            "Late Majority": 0.5,
            "Laggard": 0.3
        }
        
        if random.random() < adoption_probability[adopter_category]:
            # Set adoption timing in months (how many months after product release)
            base_timing = {
                "Innovator": (0, 1),
                "Early Adopter": (1, 3),
                "Early Majority": (3, 12),
                "Late Majority": (12, 24),
                "Laggard": (24, 48)
            }
            
            timing_range = base_timing[adopter_category]
            adoption_timing[category] = random.randint(timing_range[0], timing_range[1])
            
            # Select product preference based on budget
            product_by_price = {
                "Premium": products[0:2],
                "High": products[0:3],
                "Mid-range": products[1:4],
                "Budget": products[2:5],
                "Economy": products[3:5]
            }
            
            adoption_products[category] = random.choice(product_by_price[budget_category])
        else:
            adoption_timing[category] = -1  # -1 indicates non-adoption
            adoption_products[category] = "None"
    
    # Purchasing factors influenced by adopter category and budget
    if "Innovator" in adopter_category:
        factor_choices = ["Features", "Design", "Brand Reputation"]
    elif "Early Adopter" in adopter_category:
        factor_choices = ["Features", "Brand Reputation", "Design", "Compatibility"]
    elif "Early Majority" in adopter_category:
        factor_choices = ["Features", "Reviews", "Price", "Brand Reputation"]
    elif "Late Majority" in adopter_category:
        factor_choices = ["Price", "Reviews", "Brand Reputation", "Compatibility"]
    else:  # Laggard
        factor_choices = ["Price", "Compatibility", "Sustainability"]
        
    # Add some randomness
    available_factors = list(set(PURCHASING_FACTORS) - set(factor_choices[:2]))
    additional_factors = random.sample(available_factors, k=1)
    purchasing_factors = factor_choices[:2] + additional_factors
    
    # Technical proficiency correlates with adopter category and age
    proficiency_map = {
        "Innovator": ["Expert", "Advanced"],
        "Early Adopter": ["Advanced", "Intermediate"],
        "Early Majority": ["Intermediate"],
        "Late Majority": ["Intermediate", "Basic"],
        "Laggard": ["Basic", "Beginner"]
    }
    
    # Adjust for age
    if age < 30:
        proficiency_options = proficiency_map[adopter_category]
    elif age > 60:
        proficiency_options = [proficiency_map[adopter_category][-1]]  # Use the lowest option
    else:
        proficiency_options = proficiency_map[adopter_category]
        
    technical_proficiency = random.choice(proficiency_options)
    
    # Formal education level
    education_levels = ["High School", "Some College", "Bachelor's", "Master's", "PhD"]
    education_weights = [0.1, 0.2, 0.4, 0.2, 0.1]
    education = random.choices(education_levels, weights=education_weights, k=1)[0]
    
    # Job sector - tech related roles more likely to be early adopters
    tech_sectors = ["Technology", "Engineering", "Digital Marketing", "Data Science", "UX/UI Design"]
    non_tech_sectors = ["Finance", "Healthcare", "Education", "Retail", "Manufacturing", "Service Industry", 
                        "Government", "Legal", "Arts", "Hospitality"]
    
    if "Innovator" in adopter_category or "Early Adopter" in adopter_category:
        sector_weights = [0.7, 0.3]  # 70% chance of tech sector
    elif "Early Majority" in adopter_category:
        sector_weights = [0.5, 0.5]  # 50% chance of tech sector
    else:
        sector_weights = [0.2, 0.8]  # 20% chance of tech sector
        
    if random.choices([True, False], weights=sector_weights, k=1)[0]:
        job_sector = random.choice(tech_sectors)
    else:
        job_sector = random.choice(non_tech_sectors)
    
    # Generate location (US cities with tech hubs more likely for innovators)
    tech_hubs = ["San Francisco, CA", "Seattle, WA", "Austin, TX", "Boston, MA", "New York, NY", 
                 "Denver, CO", "San Jose, CA", "Raleigh, NC", "Atlanta, GA", "Portland, OR"]
    
    other_cities = [f"{fake.city()}, {fake.state_abbr()}" for _ in range(20)]
    
    if "Innovator" in adopter_category or "Early Adopter" in adopter_category:
        location_weights = [0.6, 0.4]  # 60% chance of tech hub
    elif "Early Majority" in adopter_category:
        location_weights = [0.3, 0.7]  # 30% chance of tech hub
    else:
        location_weights = [0.1, 0.9]  # 10% chance of tech hub
        
    if random.choices([True, False], weights=location_weights, k=1)[0]:
        location = random.choice(tech_hubs)
    else:
        location = random.choice(other_cities)
    
    # Create the user profile dictionary
    user_profile = {
        "user_id": user_id,
        "age": age,
        "gender": random.choice(["Male", "Female", "Non-binary"]),
        "location": location,
        "education": education,
        "job_sector": job_sector,
        "annual_income": annual_income,
        "adopter_category": adopter_category,
        "tech_interest": tech_interest,
        "budget_category": budget_category,
        "technical_proficiency": technical_proficiency,
        "preferred_platforms": ", ".join(preferred_platforms),
        "primary_purchasing_factor": purchasing_factors[0],
        "secondary_purchasing_factor": purchasing_factors[1] if len(purchasing_factors) > 1 else None,
        "smartphone_adoption_timing": adoption_timing["Smartphones"],
        "smartphone_preferred_product": adoption_products["Smartphones"],
        "laptop_adoption_timing": adoption_timing["Laptops"],
        "laptop_preferred_product": adoption_products["Laptops"],
        "smarthome_adoption_timing": adoption_timing["Smart Home"],
        "smarthome_preferred_product": adoption_products["Smart Home"],
        "wearable_adoption_timing": adoption_timing["Wearables"],
        "wearable_preferred_product": adoption_products["Wearables"],
        "vrar_adoption_timing": adoption_timing["VR/AR"],
        "vrar_preferred_product": adoption_products["VR/AR"]
    }
    
    # Add a marketing persona type for easier targeting
    if adopter_category in ["Innovator", "Early Adopter"] and tech_interest in ["Very High", "High"]:
        if budget_category in ["Premium", "High"]:
            persona = "Premium Tech Enthusiast"
        else:
            persona = "Value-Conscious Tech Enthusiast"
    elif adopter_category in ["Early Majority"] and tech_interest in ["High", "Moderate"]:
        if job_sector in tech_sectors:
            persona = "Practical Professional"
        else:
            persona = "Mainstream Adopter"
    elif adopter_category in ["Late Majority"]:
        persona = "Tech Pragmatist"
    else:
        persona = "Late Adopter"
    
    user_profile["persona"] = persona
    
    # Add brand loyalty metrics (0-10 scale)
    if "iPhone" in user_profile["smartphone_preferred_product"]:
        apple_loyalty = random.randint(7, 10)
    elif "None" == user_profile["smartphone_preferred_product"]:
        apple_loyalty = random.randint(0, 3)
    else:
        apple_loyalty = random.randint(1, 6)
    
    if "Samsung" in user_profile["smartphone_preferred_product"] or "Samsung" in user_profile["smarthome_preferred_product"]:
        samsung_loyalty = random.randint(7, 10)
    elif "None" in [user_profile["smartphone_preferred_product"], user_profile["smarthome_preferred_product"]]:
        samsung_loyalty = random.randint(0, 3)
    else:
        samsung_loyalty = random.randint(1, 6)
    
    user_profile["apple_loyalty"] = apple_loyalty
    user_profile["samsung_loyalty"] = samsung_loyalty
    
    # Create and add the pre-formatted text description
    # This will be added to the user_profile after it's created since it needs to reference the user_profile values
    return user_profile

def generate_dataset(num_users=NUM_USERS):
    """Generate a complete dataset of tech adoption profiles"""
    users = []
    
    for i in range(num_users):
        user_id = generate_user_id(i)
        user_profile = generate_tech_adoption_profile(user_id)
        
        # Create text description
        description = create_user_description(user_profile)
        
        # Add description to user profile
        user_profile["user_description"] = description
        
        # Add to collections
        users.append(user_profile)
    
    # Create dataframe
    df = pd.DataFrame(users)
    
    return df

def main():
    """Generate and save the technology adoption dataset"""
    print(f"Generating technology adoption dataset with {NUM_USERS} users...")
    df = generate_dataset()
    
    # Format income values as currency with cents for display
    df['formatted_income'] = df['annual_income'].apply(lambda x: f"${x:.2f}")
    
    # Save the dataset
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Dataset saved to {OUTPUT_FILE}")
    print(f"Dataset shape: {df.shape}")
    
    # Print sample statistics
    print("\nSample Statistics:")
    print(f"Adopter Categories: {df['adopter_category'].value_counts().to_dict()}")
    print(f"Tech Interest Levels: {df['tech_interest'].value_counts().to_dict()}")
    print(f"Budget Categories: {df['budget_category'].value_counts().to_dict()}")
    print(f"Technical Proficiency: {df['technical_proficiency'].value_counts().to_dict()}")
    print(f"Personas: {df['persona'].value_counts().to_dict()}")
    
    print("\nSample of 5 users with income:")
    sample_columns = ['user_id', 'age', 'adopter_category', 'formatted_income', 'tech_interest', 'persona']
    print(df[sample_columns].sample(5).to_string())
    
    print("\nSample user description:")
    print(df['user_description'].iloc[0])

if __name__ == "__main__":
    main() 