import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

def feature_engineering(data):
    selected_features = [
        'Birth_rate_crude_(per_1,000_people)',
        'Fertility_rate_total_(births_per_woman)',
        'GNI_per_capita_Atlas_method_(current_US$)',
        'Labor_force_total',
        'Life_expectancy_at_birth_total_(years)',
        'People_using_safely_managed_sanitation_services_(%_of_population)',
        'People_using_safely_managed_sanitation_services_rural_(%_of_rural_population)',
        'People_using_safely_managed_sanitation_services_urban__(%_of_urban_population)',
        'Population_total',
        'Rural_population',
        'School_enrollment_secondary_female_(%_gross)',
        'School_enrollment_tertiary_(%_gross)',
        'Urban_population',
        'Net_trade_in_goods_and_services(current_$USD)',
        'electricity_demand(in_TWh)'
    ]

    data = data[selected_features]

    features_to_transform = ['electricity_demand(in_TWh)', 'GNI_per_capita_Atlas_method_(current_US$)', 'Urban_population', 'Rural_population']

    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(data[features_to_transform])
    poly_features_df = pd.DataFrame(X_poly, columns=poly.get_feature_names_out(features_to_transform))

    interaction_features = data[['electricity_demand(in_TWh)', 'Urban_population', 'Labor_force_total']]
    X_interactions = interaction_features.copy()
    X_interactions['electricity_Urban'] = interaction_features['electricity_demand(in_TWh)'] * interaction_features['Urban_population']
    X_interactions['electricity_Labor'] = interaction_features['electricity_demand(in_TWh)'] * interaction_features['Labor_force_total']

    X_log = data[['Population_total', 'Labor_force_total']].copy()
    X_log['Population_total'] = np.log1p(X_log['Population_total'])
    X_log['Labor_force_total'] = np.log1p(X_log['Labor_force_total'])

    X_engineered = pd.concat([data, poly_features_df, X_interactions, X_log], axis=1)
    return X_engineered
