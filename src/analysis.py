def momentum(image, p, q, accepted):
    m = 0
    height, width = image.shape
    for col in range(width):
        for row in range(height):
            if accepted(image[row, col]):
                m += pow(col, q) + pow(row, p)
    return m


def central_momentum(image, p, q, accepted):
    m = 0
    m_zero = momentum(image, 0, 0, accepted)
    i = momentum(image, 1, 0, accepted) / m_zero
    j = momentum(image, 0, 1, accepted) / m_zero
    height, width = image.shape
    for col in range(width):
        for row in range(height):
            if accepted(image[row, col]):
                m += pow(col - j, q) + pow(row - i, p)
    return m


def image_invariants(image, accepted):
    M = {}
    invariants = []
    m00 = momentum(image, 0, 0, accepted)
    m01 = momentum(image, 0, 1, accepted)
    m10 = momentum(image, 1, 0, accepted)
    m11 = momentum(image, 1, 1, accepted)
    m20 = momentum(image, 2, 0, accepted)
    m02 = momentum(image, 0, 2, accepted)
    m21 = momentum(image, 2, 1, accepted)
    m12 = momentum(image, 1, 2, accepted)
    m30 = momentum(image, 3, 0, accepted)
    m03 = momentum(image, 0, 3, accepted)
    i = m10 / m00
    j = m01 / m00
    M[0] = m00
    M[1] = 0
    M[10] = 0
    M[11] = m11 - m10 * m01 / m00
    M[20] = m20 - pow(m10, 2) / m00
    M[2] = m02 - pow(m01, 2) / m00
    M[21] = m21 - 2 * m11 * i - m20 * j + 2 * m01 * pow(i, 2)
    M[12] = m12 - 2 * m11 * j - m02 * i + 2 * m10 * pow(j, 2)
    M[30] = m30 - 3 * m20 * i + 2 * m10 * pow(i, 2)
    M[3] = m03 - 3 * m02 * j + 2 * m01 * pow(j, 2)
    invariants.append((M[20] + M[2]) / pow(m00, 2))
    invariants.append((pow(M[20] - M[2], 2) + 4 * pow(M[11], 2)) / pow(m00, 4))
    invariants.append((pow(M[30] - 3 * M[12], 2) + pow(3 * M[21] - M[3], 2)) / pow(m00, 5))
    invariants.append((pow(M[30] + M[12], 2) + pow(M[21] + M[3], 2)) / pow(m00, 5))
    invariants.append(((M[30] - 3 * M[12]) * (M[30] + M[12]) * (pow(M[30] + M[12], 2) - 3 * pow(M[21] + M[3], 2))
    + (3 * M[21] - M[3]) * (M[21] + M[3]) * (3 * pow(M[30] + M[12], 2) - pow(M[21] + M[3], 2))) / pow(m00, 10))
    invariants.append(((M[20] - M[2]) * (pow(M[30] + M[12], 2) - pow(M[21] + M[3], 2)) + 4 * M[11] * (M[30] + M[12]) * (M[21] + M[3])) / pow(m00, 7))
    invariants.append((M[20] * M[2] - pow(M[11], 2)) / pow(m00, 4))
    invariants.append((M[30] * M[12] + M[21] * M[3] - pow(M[12], 2) - pow(M[21], 2)) / pow(m00, 5))
    invariants.append((M[20] * (M[21] * M[3] - pow(M[12], 2)) + M[2] * (M[3] * M[12] - pow(M[21], 2)) - M[11] * (M[30] * M[3] - M[21] * M[12])) / pow(m00, 7))
    invariants.append((pow(M[30] * M[3] - M[12] * M[21], 2) - 4 * (M[30] * M[12] - pow(M[21], 2)) * (M[3] * M[21] - M[12])) / pow(m00, 10))

    return invariants

