import os
import numpy as np
import pandas as pd
from odf.opendocument import load
from odf.table import Table, TableRow, TableCell
from odf.text import P

ods_file = "metadata/comparisons/comparisons.ods"

csv_mappings = {
    "xgboost": "metadata/comparisons/xgboost_comparison.csv",
    "lightgbm": "metadata/comparisons/lightgbm_comparison.csv",
    "catboost": "metadata/comparisons/catboost_comparison.csv",
    "elastic_tree": "metadata/comparisons/elastictree_comparison.csv",
    "elastic_net": "metadata/comparisons/elasticnet_comparison.csv",
}

def create_typed_cell(val) -> TableCell:
    """
    Creates a TableCell with the correct OpenDocument value types
    so OpenOffice Calc parses floats, integers, and booleans natively instead of text.
    """
    cell = TableCell()

    if pd.isna(val):
        return cell

    # Handle Booleans
    if isinstance(val, (bool, np.bool_)):
        cell.setAttribute('valuetype', 'boolean')
        cell.setAttribute('booleanvalue', 'true' if bool(val) else 'false')
        cell.addElement(P(text=str(val).upper()))

    # Handle Numeric Values (floats, ints, numpy numbers)
    elif isinstance(val, (int, float, np.integer, np.floating)):
        cell.setAttribute('valuetype', 'float')
        cell.setAttribute('value', str(val))
        cell.addElement(P(text=str(val)))

    # Default to Strings (text, team names, dates)
    else:
        cell.setAttribute('valuetype', 'string')
        cell.addElement(P(text=str(val)))

    return cell


def update_ods_with_csvs(ods_filepath: str, model_csv_map: dict):
    """
    Clears data in target model sheets inside an .ods file and replaces it with CSV data,
    preserving the 6th sheet (Averages/Formulas) completely.
    """
    if not os.path.exists(ods_filepath):
        print(f"[!] Target ODS file not found: '{ods_filepath}'")
        return

    doc = load(ods_filepath)
    sheets = {table.getAttribute('name'): table for table in doc.spreadsheet.getElementsByType(Table)}

    for sheet_name, csv_path in model_csv_map.items():
        if sheet_name not in sheets:
            print(f"[!] Warning: Sheet '{sheet_name}' not found in ODS. Skipping.")
            continue

        if not os.path.exists(csv_path):
            print(f"[!] Warning: CSV file '{csv_path}' not found. Skipping.")
            continue

        df = pd.read_csv(csv_path)
        table = sheets[sheet_name]

        # 1. Clear existing rows inside the target sheet
        for r in list(table.getElementsByType(TableRow)):
            table.removeChild(r)

        # 2. Add Header Row (always string text)
        header_row = TableRow()
        for col in df.columns:
            cell = TableCell()
            cell.setAttribute('valuetype', 'string')
            cell.addElement(P(text=str(col)))
            header_row.addElement(cell)
        table.addElement(header_row)

        # 3. Add Data Rows with explicit types
        for row_tuple in df.itertuples(index=False):
            data_row = TableRow()
            for val in row_tuple:
                data_row.addElement(create_typed_cell(val))
            table.addElement(data_row)

        print(f"[✓] Updated sheet '{sheet_name}' with {len(df)} typed records from '{csv_path}'")

    doc.save(ods_filepath)
    print(f"\n[🎉] Successfully updated ODS document: '{ods_filepath}'")

if __name__ == "__main__":
    update_ods_with_csvs(ods_file, csv_mappings)