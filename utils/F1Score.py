def F1Score(predict, real, frame_num, visual=False):
    # each input is dictionary, where the key is frame num, and value is:
    # array of arrays: [x, y, z, class,
    # class is: {blue, yellow, small orange, big orange}
    predict = predict[:, :2]
    real = real[:, :2]
    # threshhold in m
    threshhold = 0.4

    TP = []
    FP = []
    FN = []
    indices = []
    for i, cone in enumerate(real):
        flag = False
        for j, p_cone in enumerate(predict):
            if distance(cone, p_cone) <= threshhold:
                TP.append(cone)
                indices.append(j)
                flag = True
                break
        if not flag:
            FN.append(cone)
    for i, p_cone in enumerate(predict):
        if i not in indices:
            FP.append(p_cone)

    prediction = float(len(TP) / (len(TP) + len(FP)))
    recall = float(len(TP) / (len(TP) + len(FN)))
    f1 = float(len(TP) / (len(TP) + 0.5 * len(FN) + 0.5 * len(FP))) if (prediction !=0 or recall !=0) !=0 else 0

    return prediction*100, recall*100, f1*100


def distance(a, b):
    x = a[0] - b[0]
    y = a[1] - b[1]
    return (x ** 2 + y ** 2) ** 0.5
