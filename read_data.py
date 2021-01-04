

def read_data(file, num_in):
    inputs = []
    outputs = []
    with open(file, "r") as f:
        next(f)
        for line in f:
            separated = [float(x) for x in line.split(',')]
            inputs.append(separated[:num_in])
            outputs.append(separated[num_in:])

    return inputs, outputs


def read_extremes(file):
    with open(file, "r") as f:
        line = f.readline()
        return line.split(',')
