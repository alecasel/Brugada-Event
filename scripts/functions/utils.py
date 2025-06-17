import os
import numpy as np


def import_variables_from_yaml(yaml_file):
    """
    Import variables from a YAML file and return them as a dictionary.

    Args:
        yaml_file (str): Path to the YAML file.

    Returns:
        dict: Dictionary containing the variables from the YAML file.
    """
    import yaml

    with open(yaml_file, 'r') as file:
        variables = yaml.safe_load(file)

    return variables


def save_predictions_to_excel(lead,
                              predictions,
                              xml_file,
                              subfolder_name,
                              output_file='predizioni.xlsx'):
    """
    Save predictions to an Excel file with two sheets:
    - The first sheet contains the class corresponding to the max probability.
    - The second sheet contains the individual predictions for each class.
    """
    import pandas as pd

    if isinstance(predictions, list):
        predictions = np.array(predictions[0])

    predicted_classes = [int(np.argmax(pred, axis=0)) for pred in predictions]

    new_row_class = {
        'patient': subfolder_name,
        'xml_file': xml_file,
        'lead': lead,
    }

    # Add a column for each predicted class
    for i, predicted_class in enumerate(predicted_classes):
        new_row_class[f'predicted_class_{i}'] = predicted_class

    # Count occurrences of each class
    num_class_0 = predicted_classes.count(0)
    num_class_1 = predicted_classes.count(1)
    num_class_2 = predicted_classes.count(2)

    # Add the counts to the new_row_class dictionary
    new_row_class['num_class_0'] = num_class_0
    new_row_class['num_class_1'] = num_class_1
    new_row_class['num_class_2'] = num_class_2

    # Create detailed predictions rows - one row for each prediction
    detailed_predictions_rows = []
    for i, pred in enumerate(predictions):
        new_row_predictions = {
            'patient': subfolder_name,
            'xml_file': xml_file,
            'lead': lead,
            'prediction_index': i,
            'class_0': pred[0],
            'class_1': pred[1],
            'class_2': pred[2],
            'predicted_class': int(np.argmax(pred))
        }
        detailed_predictions_rows.append(new_row_predictions)

    if os.path.exists(output_file):
        # Read existing data
        try:
            df_class_existing = pd.read_excel(
                output_file, sheet_name='Class_Predictions')
        except ValueError:
            df_class_existing = pd.DataFrame()

        try:
            df_pred_existing = pd.read_excel(
                output_file, sheet_name='Detailed_Predictions')
        except ValueError:
            df_pred_existing = pd.DataFrame()

        # Append new data
        if not df_class_existing.empty:
            df_class = pd.concat(
                [df_class_existing, pd.DataFrame([new_row_class])],
                ignore_index=True)
        else:
            df_class = pd.DataFrame([new_row_class])

        if not df_pred_existing.empty:
            df_pred = pd.concat([df_pred_existing, pd.DataFrame(
                detailed_predictions_rows)], ignore_index=True)
        else:
            df_pred = pd.DataFrame(detailed_predictions_rows)

        # Riordina le colonne del DataFrame delle classi
        df_class = _reorder_class_columns(df_class)

        # Write both sheets
        with pd.ExcelWriter(output_file, engine='openpyxl', mode='w')\
                as writer:
            df_class.to_excel(
                writer, sheet_name='Class_Predictions', index=False)
            df_pred.to_excel(
                writer, sheet_name='Detailed_Predictions', index=False)
    else:
        # File does not exist, create new one
        df_class = pd.DataFrame([new_row_class])
        df_pred = pd.DataFrame(detailed_predictions_rows)

        # Riordina le colonne del DataFrame delle classi
        df_class = _reorder_class_columns(df_class)

        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            df_class.to_excel(
                writer, sheet_name='Class_Predictions', index=False)
            df_pred.to_excel(
                writer, sheet_name='Detailed_Predictions', index=False)


def _reorder_class_columns(df):
    """
    Riordina le colonne del DataFrame in modo che:
    - xml_file e lead vengano per primi
    - tutte le colonne predicted_class_x vengano dopo
    - tutte le colonne num_class_x vengano alla fine
    """
    # Ottieni tutte le colonne del DataFrame
    all_columns = df.columns.tolist()
    
    # Separa le colonne in categorie
    base_columns = ['patient', 'xml_file', 'lead']
    predicted_columns = [col for col in all_columns if col.startswith('predicted_class_')]
    num_columns = [col for col in all_columns if col.startswith('num_class_')]
    other_columns = [col for col in all_columns if col not in base_columns + predicted_columns + num_columns]
    
    # Ordina le colonne predicted_class e num_class numericamente
    predicted_columns.sort(key=lambda x: int(x.split('_')[-1]))
    num_columns.sort(key=lambda x: int(x.split('_')[-1]))
    
    # Crea l'ordine finale delle colonne
    ordered_columns = base_columns + other_columns + predicted_columns + num_columns
    
    # Riordina il DataFrame
    return df[ordered_columns]