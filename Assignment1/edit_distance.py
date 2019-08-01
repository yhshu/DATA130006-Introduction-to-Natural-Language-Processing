import Levenshtein


def edit_distance(str1, str2, m, n):
    """
    A Naive recursive Python program to find minimum number operations to convert str1 to str2

    Reference:
    https://www.geeksforgeeks.org/edit-distance-dp-5/
    """

    # If first string is empty, the only option is to
    # insert all characters of second string into first
    if m == 0:
        return n

        # If second string is empty, the only option is to
    # remove all characters of first string
    if n == 0:
        return m

        # If last characters of two strings are same, nothing
    # much to do. Ignore last characters and get count for
    # remaining strings.
    if str1[m - 1] == str2[n - 1]:
        return self.edit_distance(str1, str2, m - 1, n - 1)

        # If last characters are not same, consider all three
    # operations on last character of first string, recursively
    # compute minimum cost for all three operations and take
    # minimum of three values.
    return 1 + min(self.edit_distance(str1, str2, m, n - 1),  # Insert
                   self.edit_distance(str1, str2, m - 1, n),  # Remove
                   self.edit_distance(str1, str2, m - 1, n - 1)  # Replace
                   )


def _Damerau_Levenshtein_edit_distance(str1, str2):
    """
    Calculate Damerau-Levenshtein Edit Distance for two string

    Reference:
    https://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance

    :param str1: string 1
    :param str2: string 2
    :return: the Damerau-Levenshtein Edit Distance between str1 and str2
    """
    str1 = '@' + str1
    str2 = '@' + str2
    len1 = len(str1)
    len2 = len(str2)
    dis = [[0] * len2 for _ in range(len1)]
    for i in range(len1):
        for j in range(len2):
            dis[i][0] = i
            dis[0][j] = j

    for i in range(len1):
        for j in range(len2):
            if i == 0 or j == 0:
                continue  # dis[0][0] = 0

            t = [0] * 4
            t[0] = dis[i - 1][j] + 1
            t[1] = dis[i][j - 1] + 1
            if str1[i] != str2[j]:
                t[2] = dis[i - 1][j - 1] + 1
            else:
                t[2] = dis[i - 1][j - 1]
            if str1[i] == str2[j - 1] and str1[i - 1] == str2[j]:  # transposition of two adjacent characters
                t[3] = dis[i - 1][j - 1] - 1
            if t[3] != 0:
                dis[i][j] = min(t[0:4])
            else:
                dis[i][j] = min(t[0:3])
    return dis[len1 - 1][len2 - 1]


def Damerau_Levenshtein_edit_distance(str1, str2):
    if len(str1) == len(str2) and Levenshtein.hamming(str1, str2) == 2:
        return _Damerau_Levenshtein_edit_distance(str1, str2)
    return Levenshtein.distance(str1, str2)  # insertion, deletion, substitution
