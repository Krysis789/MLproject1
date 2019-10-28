def replace_with_mean(data, replacedValue):
    means = []
    def replace_one_col(x):
        m = np.mean(x[x != replacedValue])
        x[x == replacedValue] = m
        means.append(m)
        return x
    return np.apply_along_axis(replace_one_col, 0, np.copy(data)), means

def replace_with_values(data, values, replacedValue):
    result = np.copy(data)
    for i in range(np.shape(data)[1]):
        for j in range(np.shape(data)[0]):
            if result[j, i] == replacedValue:
                result[j, i] = values[i]
    return result
            
def centralize_data(data):
    return np.apply_along_axis(lambda x: x - np.mean(x), 0, np.copy(data))

def subtract_values(data, values):
    result = np.copy(data)
    for i in range(np.shape(data)[1]):
        for j in range(np.shape(data)[0]):
            result[j, i] = result[j, i] - values[i]
    return result


def scale_data(data):
    deviations = np.apply_along_axis(lambda x: np.std(x), 0, np.copy(data))
    deviations[deviations == 0] = 1
    return np.apply_along_axis(lambda x: x / np.std(x) if np.std(x) != 0 else x, 0, np.copy(data)), deviations

def divide_by_values(data, values):
    result = np.copy(data)
    for i in range(np.shape(data)[1]):
        for j in range(np.shape(data)[0]):
            result[j, i] = result[j, i] / values[i]
    return result

def add_constant_term(data):
    a = np.ones((np.shape(data)[0], np.shape(data)[1] + 1))
    a[:, :-1] = data
    return a

def add_missing_value_info(data, missingValue):
    a = np.zeros((np.shape(data)[0], np.shape(data)[1] + 1))
    a[:, :-1] = data
    a[:, np.shape(a)[1] - 1] = np.apply_along_axis(lambda x: np.sum(x == missingValue), 1, data)
    return a