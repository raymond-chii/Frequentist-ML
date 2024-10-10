import csv


def convert_txt_to_csv(input_file, output_file):
    with open(input_file, "r") as txt_file, open(
        output_file, "w", newline=""
    ) as csv_file:
        csv_writer = csv.writer(csv_file)

        # Read and write the header
        header = next(txt_file).strip().split(",") # Skip the 'row.names' column

        # Read and write the data rows
        for line in txt_file:
            data = line.strip().split(",")
            csv_writer.writerow(data)  # Skip the first column (row number)


if __name__ == "__main__":
    input_file = "datatest.txt"
    output_file = "OccupancyDetection.csv"
    convert_txt_to_csv(input_file, output_file)
    print(f"Conversion complete. CSV file saved as {output_file}")
