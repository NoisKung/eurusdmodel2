import csv
from datetime import datetime

# define the input and output file paths
#input_file = 'rates2.csv'
#output_file = 'rates2_data_sorted.csv'

# define a function to convert a datetime string to a datetime object
def convert_datetime_string_to_datetime(datetime_string):
    return datetime.strptime(datetime_string, '%Y-%m-%d %H:%M:%S')

def sort_file(input_file, output_file):
# read the input CSV file and sort the rows based on the datetime column
    with open(input_file, 'r') as f_input, open(output_file, 'w', newline='') as f_output:
        csv_reader = csv.reader(f_input)
        csv_writer = csv.writer(f_output)
    
        # write the header row to the output CSV file
        header_row = next(csv_reader)
        csv_writer.writerow(header_row)
    
        # sort the rows based on the datetime column
        sorted_rows = sorted(csv_reader, key=lambda row: convert_datetime_string_to_datetime(row[0]))
    
        # write the sorted rows to the output CSV file
        for row in sorted_rows:
            csv_writer.writerow(row)