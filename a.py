def longest_common_substring(str1, str2):
    # Create a 2D array to store lengths of longest common suffixes
    m, n = len(str1), len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    longest = 0  # Length of longest common substring
    lcs_end_pos = 0  # End position of LCS in str1

    # Build the dp array
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > longest:
                    longest = dp[i][j]
                    lcs_end_pos = i
            else:
                dp[i][j] = 0

    # Return the longest common substring
    return str1[lcs_end_pos - longest:lcs_end_pos]

def identify_belonging(passages, sentences):
    results = []

    for i, sentence in enumerate(sentences):
        max_length = 0
        belonging_passage = None
        # Compare each sentence with each passage
        for passage in passages:
            lcs = longest_common_substring(passage, sentence)
            lcs_length = len(lcs)
            # Check if the current LCS is the longest found so far
            if lcs_length > max_length:
                max_length = lcs_length
                belonging_passage = passage

        # Append the result as a tuple (sentence, index of the passage)
        results.append((i, passages.index(belonging_passage)))

    return results

# Example usage
passages = ["This is a passage about science.", "Here we discuss the art of painting."]
sentences = ["This passage discusses science topics.", "The art discussed involves painting."]
result = identify_belonging(passages, sentences)
print(result)
