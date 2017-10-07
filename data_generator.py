import functions
import numpy as np
import matplotlib.pyplot as plt
import sys
import random
import time
import csv
import argparse

############## Data generators for logistic regression #########

x_static = np.array([[ 17.84591693, -20.31212513, -5.34325409],
 [ 29.95789925,-32.3866854, -8.41519701],
 [ 15.63310873,-18.08745614 , -4.7878777 ],
 [ 18.2520592 ,-20.5402138  , -5.39036512],
 [ 21.72654606,-24.43094582 , -6.40960318],
 [ 30.9639126 ,-33.52595032 , -8.7104181 ],
 [ 27.44394365,-29.66375317 , -7.70687231],
 [ 33.43986863,-36.10186048 , -9.36801723],
 [ 26.23476558,-28.73271915 , -7.48212717],
 [ 11.99775663,-13.49622317 , -3.55432812]])

################################################################

def get_random_dataset(filename, entries):
    count = 0
    with open(filename, 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',',
                            quotechar='|')
        for row in spamreader:
            count += 1

    dataset = []
    pass_0_values = [random.randint(1, count/2) for i in range(entries)]
    pass_1_values = [random.randint(count/2 +1,count) for i in range(entries)]
    i = 0
    with open(filename, 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',',
                            quotechar='|')
        for row in spamreader:
            i += 1
            if i in pass_0_values or i in pass_1_values:
                dataset.append([float(j)for j in row])
    return dataset


def generate_random_vector(total, dim, type='uniform'):
    if type=='uniform':
        return np.random.rand(total, dim)
    elif type=='normal':
        return np.random.randn(total, dim)
    else:
        print("You entered something wierd.\n Returning empty array")
        return np.array([])

def append_static(vector, static):
    new_array = []
    for vec in vector:
        a = list(vec)
        a.append(static)
        new_array.append(a)
    return np.array(new_array)

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Dataset generator.')
    parser.add_argument('--savefile', dest='savefile', type=str,
                        help='name to the file to save')
    parser.add_argument('--type', dest='type', type=str,
                        help='type of data to be generated')
    parser.add_argument('--dim', dest='dim', type=int, default=2,
                        help='dimension of the data being generated')
    parser.add_argument('--num', dest='num', nargs='+', type=int,
                        help='2-D number vector. For e.g. 10 true and 20 false samples will be written as 10 20')
    args = parser.parse_args()

    u_entries = args.num
    savefile = args.savefile
    data_type = args.type

    # TODO(PARTH) : ADD DESCRIPTION OF OF THE COLUMNS IN THE EXCEL SHEET
    if data_type == 'logistic':
        total_entries = sum(u_entries)*10
        # TODO(PARTH) : Have to work on the splitting up of the sizes of 1 and 0 labelled data. Still not implemented yet.
        # TODO(PARTH) : Random dimension feature is not added in logistic regression model. Will get implemented in future releases.
        contenders = generate_random_vector(total_entries, dim=args.dim-1)
        arrays = append_static(contenders, 1)

        pass_1 = []
        count_1 = 0
        pass_0 = []
        count_0 = 0

        for arr in arrays:
            decider = sum([functions.LogisticRegressionClass.event_probability(arr, evaluater) for evaluater in x_static])/len(x_static)
            if decider < 0.001 and count_0 < 1000:
                pass_0.append(arr)
                count_0 += 1
            elif decider > 0.95 and count_1 < 1000:
                pass_1.append(arr)
                count_1 += 1
            elif count_0 + count_1 == 2000:
                break
            sys.stdout.write("\r{Pass 0 : %d Pass 1 : %d}"%(len(pass_0), len(pass_1)))

            sys.stdout.flush()

        with open(savefile, 'wb') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for entry in pass_0:
                new_entry = list(entry[:-1])
                new_entry.append(-1)
                spamwriter.writerow(new_entry)

            for entry in pass_1:
                new_entry = list(entry[:-1])
                new_entry.append(1)
                spamwriter.writerow(new_entry)

    elif data_type == 'linear':

        x_bar = sum(generate_random_vector(10, dim=(args.dim+1), type='normal'))

        total_entries = sum(u_entries)

        contenders = generate_random_vector(total_entries, dim=args.dim, type='normal')
        contenders = 5*contenders

        arrays = append_static(contenders, 1)

        dataset = []
        for arr in arrays:
            dataset.append(list(arr))
            dataset[-1].append(np.dot(arr, x_bar) + np.random.normal())

        with open(savefile, 'wb') as csvfile:
            spamwriter = csv.writer(csvfile, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
            for entry in dataset:
                new_entry = list(entry[:-2])
                new_entry.append(entry[-1])
                spamwriter.writerow(new_entry)