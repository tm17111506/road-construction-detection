import os

path = '/media/tma1/Elements1/RECORDS/'
base_format = 'python3 read_bag.py -f /media/tma1/Elements1/RECORDS/{}/ -o /usr0/tma1/datasets/bus_edge/{}/ -c 3'

for f in os.listdir(path):
    date_folder = f.split('/')[-1]
    print(base_format.format(date_folder, date_folder))