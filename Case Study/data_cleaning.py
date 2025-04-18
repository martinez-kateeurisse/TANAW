import os
import pandas as pd
from datetime import datetime
import re


def clean_data(file_path):
    df = pd.read_csv(file_path, header=None)
    cleaned_files_directory = os.path.join(os.path.dirname(__file__), 'cleaned_files')
    os.makedirs(cleaned_files_directory, exist_ok=True)

    # Keywords that typically appear in the header row
    potential_headers = ['Region', 'Kindergarten', 'Grade 1', 'Grade 2', 'G1', 'G2']
    header_row_index = None

    for idx, row in df.iterrows():
        row_values = row.astype(str).str.lower().tolist()
        if all(any(keyword.lower() in cell for cell in row_values) for keyword in ['region']) and \
           any(any(grade.lower() in cell for cell in row_values) for grade in ['kindergarten', 'grade 1', 'g1']):
            header_row_index = idx
            break

    if header_row_index is None:
        raise ValueError("Could not find a valid header row.")

    df.columns = df.iloc[header_row_index]
    df_cleaned = df.iloc[header_row_index + 1:].reset_index(drop=True)

    df_cleaned = df_cleaned.apply(pd.to_numeric, errors='ignore')

    # Check if it's a school-level dataset (standard) or just region-level
    is_school_level = 'School Name' in df_cleaned.columns and 'BEIS School ID' in df_cleaned.columns

    if is_school_level:
        # ========== STANDARD SCHOOL-LEVEL CLEANING ==========
        special_cases = {
            r'\bES\b': 'ELEMENTARY SCHOOL',
            'E/S': 'ELEMENTARY SCHOOL',
            r'\bELEM.\b': 'ELEMENTARY SCHOOL',
            r'\bNHS\b': 'NATIONAL HIGH SCHOOL',
            r'\bHS\b': 'HIGH SCHOOL',
            r'\bCES\b': 'CENTRAL ELEMENTARY SCHOOL',
            r'\bSCH.\b': 'SCHOOL',
            'Incorporated': 'INC.',
            r'\bMEM.\b': 'MEMORIAL',
            r'\bCS\b': 'CENTRAL SCHOOL',
            r'\bPS\b': 'PRIMARY SCHOOL',
            'P/S': 'PRIMARY SCHOOL',
            r'\bLC\b': 'LEARNING CENTER',
            'BARANGAY': 'BRGY. ',
            'POBLACION': 'POB. ',
            'STREET': 'ST. ',
            'BUILDING': 'BLDG. ',
            'BLOCK': 'BLK. ',
            'PUROK': 'PRK. ',
            'AVENUE': 'AVE. ',
            'ROAD': 'RD. ',
            'PACKAGE': 'PKG. ',
            'PHASE': 'PH. ',
            r'\s*,\s*': ', ',
            r'\s{2,}': ' '
        }

        columns_to_format = ['School Name', 'Street Address', 'Province', 'Municipality', 'Barangay']
        df_cleaned[columns_to_format] = df_cleaned[columns_to_format].apply(
            lambda x: x.str.replace('#', '', regex=False)
                        .str.replace(r'^[-:]', '', regex=True)
                        .str.strip()
                        .str.upper()
                        .replace(special_cases, regex=True)
        )

        columns_to_format = ['Street Address', 'Barangay']
        df_cleaned[columns_to_format] = (
            df_cleaned[columns_to_format]
            .replace(['N/A', 'N.A.', 'N / A', 'NA', 'NONE', 'NULL', 'NOT APPLICABLE', '', '0', '_', '=', '.', '-----'], pd.NA)
            .replace(r'^[\s\W_]+$', pd.NA, regex=True)
            .fillna("UNKNOWN")
        )

        non_enrollment_cols = [
            'Region', 'Division', 'District', 'BEIS School ID', 'School Name',
            'Street Address', 'Province', 'Municipality', 'Legislative District',
            'Barangay', 'Sector', 'School Subclassification', 'School Type', 'Modified COC'
        ]

        enrollment_cols = [col for col in df_cleaned.columns if col not in non_enrollment_cols]
        max_threshold = 5000

        unrealistic_data = df_cleaned[(
            df_cleaned[enrollment_cols] < 0).any(axis=1) |
            (df_cleaned[enrollment_cols] % 1 != 0).any(axis=1) |
            (df_cleaned[enrollment_cols] > max_threshold).any(axis=1)
        ]

        df_cleaned = df_cleaned.drop(unrealistic_data.index)

        df_cleaned.columns = [standardize_column_name(col) for col in df_cleaned.columns]
        cleaned_filename = f"cleaned_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        cleaned_path = os.path.join(cleaned_files_directory, cleaned_filename)

        if isinstance(df_cleaned, pd.DataFrame):
            df_cleaned.to_csv(cleaned_path, index=False)
            return cleaned_path 
        else:
            raise ValueError("DataFrame was not cleaned properly.")

    else:
        # ========== REGIONAL-LEVEL CLEANING ==========

        non_enrollment_cols = ['Region']
        gender_indicators = ['male', 'female', 'm', 'f']
        header_row_index = None

        for idx, row in df.iterrows():
            row_values = row.astype(str).str.lower().tolist()
            if 'region' in row_values:
                header_row_index = idx
                break

        if header_row_index is None:
            raise ValueError("Could not find a valid header row.")

        df_trimmed = df.iloc[header_row_index:].reset_index(drop=True)

        grade_row = df_trimmed.iloc[0].tolist()
        gender_row = df_trimmed.iloc[1].tolist()

        # Build the new headers
        new_columns = []
        last_valid_grade = None
        for i in range(len(gender_row)):
            grade = str(grade_row[i]).strip() if i < len(grade_row) else ""
            gender = str(gender_row[i]).strip() if i < len(gender_row) else ""

            if grade.lower() == 'region':
                new_columns.append('Region')
                continue

            if grade and grade.lower() != 'nan':
                last_valid_grade = grade
            elif not grade or grade.lower() == 'nan':
                grade = last_valid_grade

            if gender.lower() in gender_indicators:
                new_columns.append(f"{grade} {gender.title()}")
            else:
                new_columns.append(grade)

        # Get the actual data rows after headers
        df_data = df_trimmed.iloc[2:].reset_index(drop=True)

        if df_data.shape[1] != len(new_columns):
            print("⚠️ Warning: Column mismatch! Headers and data columns may not align.")

        df_data.columns = new_columns

        # Drop completely empty rows
        df_data = df_data.dropna(how='all')
        df_data = df_data.replace('-', '0')
        df_cleaned = df_data

        # Save cleaned file
        df_cleaned.columns = [standardize_column_name(col) for col in df_cleaned.columns]
        cleaned_filename = f"cleaned_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        cleaned_path = os.path.join(cleaned_files_directory, cleaned_filename)
        df_cleaned.to_csv(cleaned_path, index=False)
        return cleaned_path
    
def standardize_column_name(col):
    col = str(col).strip().upper()
    col = col.replace('K MALE', 'KINDERGARTEN MALE').replace('K FEMALE', 'KINDERGARTEN FEMALE')

    # Expand shorthand G1 to GRADE 1, G2 to GRADE 2, etc.
    col = re.sub(r'\bG(\d{1,2})\b', lambda m: f'GRADE {int(m.group(1))}', col)
    
    # Normalize Grade levels like "GRADE 1 MALE", "GRADE 1 (STEM) MALE", etc.
    col = re.sub(r'\bKINDERGARTEN\b', 'KINDERGARTEN', col)
    col = re.sub(r'\bGRADE\b\s*(\d{1,2})', lambda m: f'GRADE {int(m.group(1))}', col)

    # Move gender to the end if it appears earlier
    col = re.sub(r'\b(MALE|FEMALE)\b\s*$', r'\1', col)
    col = re.sub(r'^(MALE|FEMALE)\s+', '', col)

    # Remove parentheses, unify track/strand naming
    col = re.sub(r'\((.*?)\)', r'\1', col)  
    col = re.sub(r'\s{2,}', ' ', col)  
    col = col.strip()

    return col