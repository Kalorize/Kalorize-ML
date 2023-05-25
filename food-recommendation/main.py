import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors
import pickle

# line 9 - 21 harus di initiate
df = pd.read_csv("cleaned_recipes.csv")

min_max_scaler = MinMaxScaler()
dataset_ekstrak = df.iloc[:, 6:15]

dataset_fit = min_max_scaler.fit_transform(dataset_ekstrak)

model = pickle.load(
    open('knn.pkl', 'rb'))

indices = model.kneighbors(dataset_fit, return_distance=False)

all_calories = list(df['Calories'].values)


def closest(all_calories, input):
    return all_calories[min(range(len(all_calories)), key=lambda i: abs(all_calories[i]-input))]


def calculate_bmr(gender, weight, height, age):
    if gender == 'male':
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    elif gender == 'female':
        bmr = 10 * weight + 6.25 * height - 5 * age - 161
    else:
        raise ValueError("Invalid gender. Please choose 'male' or 'female'.")
    return bmr


def calculate_total_calories(bmr, activity_level, target):
    activity_levels = {
        'easy': 1.375,
        'medium': 1.55,
        'hard': 1.725,
        'extreme': 1.9
    }
    targets = {
        'reduce_weight': 0.8,
        'increase_muscle': 1.2,
        'be_healthy': 1
    }
    maintain_calories = bmr * activity_levels[activity_level]
    total_calories = maintain_calories * targets[target]
    return total_calories


# Ubah return ke JSON
def recommendation(gender, weight, height, age, activity_level, target):
    bmr = calculate_bmr(gender, weight, height, age)
    total_calories = calculate_total_calories(bmr, activity_level, target)

    breakfast_calories = total_calories * 0.35
    lunch_calories = total_calories * 0.40
    dinner_calories = total_calories * 0.25

    breakfast_closest_food = closest(all_calories, breakfast_calories)
    lunch_calories_food = closest(all_calories, lunch_calories)
    dinner_calories_food = closest(all_calories, lunch_calories)

    breakfast_food_code = all_calories.index(breakfast_closest_food)
    lunch_food_code = all_calories.index(lunch_calories_food)
    dinner_food_code = all_calories.index(dinner_calories_food)

    breakfast_similarity = df.iloc[indices[breakfast_food_code]]
    lunch_similarity = df.iloc[indices[lunch_food_code]]
    dinner_similarity = df.iloc[indices[dinner_food_code]]

    return breakfast_similarity, lunch_similarity, dinner_similarity


gender = 'male'
weight = 60
height = 160
age = 20
activity_level = 'medium'
target = 'reduce_weight'

breakfast, lunch, dinner = recommendation(
    gender, weight, height, age, activity_level, target)

print(breakfast)
print(lunch)
print(dinner)
