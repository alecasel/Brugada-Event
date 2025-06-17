from scripts.functions.utils import load_excel_data, save_results
from scripts.functions.prediction import process_multilevel_aggregation


def main():

    file_path = input("Inserisci il percorso del file Excel: ")

    print("Caricamento dati...")
    class_predictions, _ = load_excel_data(file_path)

    if class_predictions is not None:
        print("\nColonne disponibili in Class_Predictions:")
        print(class_predictions.columns.tolist())

        # Verifica che le colonne necessarie esistano
        required_cols = ['patient', 'xml_file', 'lead',
                         'num_class_0', 'num_class_1', 'num_class_2']
        missing_cols = [
            col for col in required_cols
            if col not in class_predictions.columns]

        if missing_cols:
            print(f"Errore: Colonne mancanti: {missing_cols}")
            return

        # Processa l'aggregazione multi-livello
        results = process_multilevel_aggregation(class_predictions)

        # Salva risultati
        save_results(results)


if __name__ == "__main__":
    main()
