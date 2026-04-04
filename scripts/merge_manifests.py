import csv
from pathlib import Path

def merge():
    main_path = Path('data/processed/financial_asr_manifest.csv')
    hindi_path = Path('data/processed/hindi_financial/hindi_financial_manifest.csv')
    out_path = Path('data/processed/combined_manifest.csv')

    rows = []
    
    if main_path.exists():
        with main_path.open('r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Ensure all 5 fields
                if 'source' not in row: row['source'] = 'english_dataset'
                if 'duration_seconds' not in row: row['duration_seconds'] = '5.000'
                rows.append(row)
            
    if hindi_path.exists():
        with hindi_path.open('r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'source' not in row: row['source'] = 'synthetic_hindi'
                if 'duration_seconds' not in row: row['duration_seconds'] = '3.000'
                rows.append(row)
            
    if rows:
        fields = ['audio_path', 'text', 'language', 'source', 'duration_seconds']
        with out_path.open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for r in rows:
                writer.writerow({f: r.get(f, '') for f in fields})
            
    print(f"Merged {len(rows)} total rows to {out_path}")

if __name__ == '__main__':
    merge()
