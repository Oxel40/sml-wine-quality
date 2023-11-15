import os


def generate_wine(quality,
        fixed_acidity_mean,
        fixed_acidity_std,
        volatile_acidity_mean,
        volatile_acidity_std,
        citric_acid_mean,
        citric_acid_std,
        residual_sugar_mean,
        residual_sugar_std,
        chlorides_mean,
        chlorides_std,
        free_sulfur_dioxide_mean,
        free_sulfur_dioxide_std,
        total_sulfur_dioxide_mean,
        total_sulfur_dioxide_std,
        density_mean,
        density_std,
        ph_mean,
        ph_std,
        sulphates_mean,
        sulphates_std,
        alcohol_mean,
        alcohol_std):
    """
    Returns a single wine as a single row in a DataFrame
    """
    import pandas as pd
    import random

    df = pd.DataFrame({ 'type': [random.choice([0, 0, 0, 1])],
                       'fixed_acidity': [random.gauss(fixed_acidity_mean, fixed_acidity_std)],
                       'volatile_acidity': [random.gauss(volatile_acidity_mean, volatile_acidity_std)],
                       'citric_acid': [random.gauss(citric_acid_mean, citric_acid_std)],
                       'residual_sugar': [random.gauss(residual_sugar_mean, residual_sugar_std)],
                       'chlorides': [random.gauss(chlorides_mean, chlorides_std)],
                       'free_sulfur_dioxide': [random.gauss(free_sulfur_dioxide_mean, free_sulfur_dioxide_std)],
                       'total_sulfur_dioxide': [random.gauss(total_sulfur_dioxide_mean, total_sulfur_dioxide_std)],
                       'density': [random.gauss(density_mean, density_std)],
                       'ph': [random.gauss(ph_mean, ph_std)],
                       'sulphates': [random.gauss(sulphates_mean, sulphates_std)],
                       'alcohol': [random.gauss(alcohol_mean, alcohol_std)]
                      })
    df['quality'] = quality
    return df


def get_random_wine():
    """
    Returns a DataFrame containing one random wine
    """
    import pandas as pd
    import random

    quality = round(random.gauss(5.82, 0.87))
    wine_df = generate_wine(quality, 
                            7.22, 1.3,
                            0.34, 0.16,
                            0.32, 0.15,
                            5.44, 4.76,
                            0.06, 0.04,
                            30.5, 17.7,
                            116,  56.5,
                            0.99, 0,
                            3.22, 0.16,
                            0.53, 0.15,
                            10.5, 1.19)

    return wine_df


def main():
    import hopsworks
    import pandas as pd

    project = hopsworks.login()
    fs = project.get_feature_store()

    wine_fg = fs.get_feature_group(name="wine",version=1)
    for _ in range(5):
        wine_fg.insert(get_random_wine())


if __name__ == "__main__":
    main()