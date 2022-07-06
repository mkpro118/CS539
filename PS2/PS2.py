# Cell 4
# Setup for Egg types
class Egg:
    '''
    This class represents each type of egg
    and their attributes, namely,
    | label : str | The label of this type of egg
    | weight: int | The weight of this type of egg
    | count : int | The number of eggs of this type
    '''
    def __init__(self, label: str, weight: int, count: int):
        self.label = label
        self.weight = weight
        self.count = count

eggs = [
    Egg('Large', 1, 15), Egg('Large', 2, 40),
    Egg('Large', 3, 10), Egg('Large', 4, 5),
    Egg('Jumbo', 1, 0), Egg('Jumbo', 2, 8),
    Egg('Jumbo', 3, 17), Egg('Jumbo', 4, 5),
]

# Cell 5
# Setting up functions to compute posterior probability

# Function to compute likelihood
def likelihood(A:int, B:str) -> float:
    '''
    Params:
        | A: int | The weight of the egg
        | B: str | The label of the egg
    Returns:
        float: likelihood
    
    This function computes and returns the likelihood of A given B
    using the formula
    P(A|B) = P(A and B) / P(B)
    '''
    global eggs
    total_number_of_eggs_with_label_B = sum(
        (egg.count for egg in eggs if egg.label == B)
    )
    eggs_with_label_A_and_weight_B = sum(
        (egg.count for egg in eggs if egg.weight == A and egg.label == B)
    )
    return eggs_with_label_A_and_weight_B / total_number_of_eggs_with_label_B

# Function to compute prior probabilites
def pr_label(A: str) -> float:
    '''
    Params:
        | A: str | The label of the egg
    Returns:
        float: prior probability of an egg with label A
    
    This function computes and returns the prior probablity
    of label A using the formula
    P(A) = Number of eggs with label A / Total number of eggs
    '''
    global eggs
    total_number_of_eggs = sum((egg.count for egg in eggs))
    number_of_eggs_with_label_A = sum((egg.count for egg in eggs if egg.label == A))
    return number_of_eggs_with_label_A / total_number_of_eggs

# Function to compute prior probabilites
def pr_weight(A: int) -> float:
    '''
    Params:
        | A: str | The weight of the egg
    Returns:
        float: prior probability of an egg with weight A
    
    This function computes and returns the prior probablity
    of weight A using the formula
    P(A) = Number of eggs with weight A / Total number of eggs
    '''
    global eggs
    total_number_of_eggs = sum((egg.count for egg in eggs))
    number_of_eggs_with_weight_A = sum((egg.count for egg in eggs if egg.weight == A))
    return number_of_eggs_with_weight_A / total_number_of_eggs

# Function to compute posterior probability
def posterior_probability(A: str, B: int):
    '''
    Params:
        | A: str | The label of the egg
        | B: int | The weight of the egg
    Returns:
        float: posterior probability of an egg with label A and weight B
    
    This function computes and returns the posterior probablity
    of weight A using the formula
    P(A|B) = (Pr(A) / Pr(B)) * L(B|A)
    P  is the Posterior Probability
    Pr is the Prior Probability
    L  is the Likelihood
    '''
    return (pr_label(A) / pr_weight(B)) * likelihood(B, A)

# Cell 6
# Computing the posterior probabilites
posterior_probabilities = dict(
    [((A, B), posterior_probability(A, B)) for A in ['Large', 'Jumbo'] for B in range(1,5)]
)
for (Label, Weight), po_p in posterior_probabilities.items():
    print(f'P({Label=} | {Weight=}g) = {po_p:.5f}')

# Cell 9
# Setup
def p_misclassification(confusion_matrix: list[list[float]]) -> float:
    '''
    Params:
        | confusion_matrix: list[list[float]] | The confusion matrix
    Returns:
        float: The probability of misclassification
    
    This function computes the probability of misclassification of the
    given confusion matrix using the formula,
    P(misclassification) = sum(off-diagonal elements) / sum(all elements)
    '''
    sum_of_all_elements = sum((sum(i) for i in confusion_matrix))
    # Sum of off diagonal elements is equal to,
    # sum of all elements - sum of diagonal elements
    sum_of_off_diagonal_elements = sum_of_all_elements - sum(
        (confusion_matrix[i][i] for i in range(len(confusion_matrix)))
    )
    return sum_of_off_diagonal_elements / sum_of_all_elements

confusion_matrix = [
    [0.873, 0.405],
    [0.127, 0.595],
]

print(f'P{{Misclassification}} = {p_misclassification(confusion_matrix)}')

