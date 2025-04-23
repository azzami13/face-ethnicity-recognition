import os
import csv

def create_metadata_csv(dataset_dir="dataset", output_csv="dataset/metadata.csv"):
    with open(output_csv, mode='w', newline='') as csvfile:
        fieldnames = ['image_path', 'name', 'ethnicity', 'expression', 'angle', 'lighting']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

        # Loop melalui suku
        for suku in os.listdir(dataset_dir):
            suku_path = os.path.join(dataset_dir, suku)
            
            if os.path.isdir(suku_path):
                # Loop melalui nama individu dalam suku
                for nama in os.listdir(suku_path):
                    nama_path = os.path.join(suku_path, nama)

                    if os.path.isdir(nama_path):
                        # Loop melalui gambar individu
                        for file in os.listdir(nama_path):
                            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                                image_path = os.path.relpath(os.path.join(nama_path, file), dataset_dir)
                                
                                # Assign nilai default untuk kolom-kolom lain (bisa diubah nanti)
                                writer.writerow({
                                    'image_path': image_path,
                                    'name': nama,
                                    'ethnicity': suku,
                                    'expression': 'Unknown',  # Default, bisa update nanti
                                    'angle': 'Front',  # Default, bisa update nanti
                                    'lighting': 'Normal'  # Default, bisa update nanti
                                })

# Panggil fungsi untuk membuat CSV
create_metadata_csv()
