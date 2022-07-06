# Cell 2
# Setup

# Tables of the egg's features and labels
egg_table = {
    'Large': [3, 8, 6, 2, 0,],
    'Jumbo': [0, 1, 5, 9, 6,],
}

class Egg:
    '''
    This class represents each type of egg
    and their attributes, namely,
    | label  : str | The label of this type of egg
    | feature: int | The feature of this type of egg
    | count  : int | The number of eggs of this type
    '''
    def __init__(self, label: str, feature: int, count: int):
        self.label = label
        self.feature = feature
        self.count = count
    
    def __repr__(self):
        return f'{self.label} | {self.feature} | {self.count}'
        

# List of Egg objects
egg_list = []
for label, features in egg_table.items():
    for feature, count in enumerate(features, 1):
        egg_list.append(Egg(label, feature, count))


# Helper functions


# maps the input to it's count values
c_map = lambda x: x.count


# calculate the prior probabilities
def pr(pred1: callable, pred2: callable = None) -> float:
    '''
    Params:
        | pred1 : callable | First condition to check
        | pred2 : callable | Optional second condition
    Returns:
        float: The prior probability given the predicates
    
    This function computes the prior probabilty, given the predicates
    
    It has two signatures,
    pr(pred1) => computes prior probability of pred1 and
                 defaults pred2 to always be True

    pr(pred1, pred2) => computes prior probability of pred1 given
                        pred2 i.e, Pr{pred1 | pred2}
    '''
    # Default pred2 to always return true if pred2 is not given
    if pred2 is None:
        pred2 = lambda x: True
    
    # Combined predicate filter key
    key = lambda x: pred1(x) and pred2(x)
    
    # The numerator
    n = sum(map(c_map, filter(key, egg_list)))
    
    # The denominator
    d = sum(map(c_map, filter(pred2, egg_list)))
    
    # Numerator / denominator
    return n / d

def pos_pr(pred1: callable, pred2: callable) -> float:
    '''
    Params:
        | pred1 : callable | First condition
        | pred2 : callable | Second condition
    Returns:
        float: The posterior probability given the predicates
    
    This function computes the posterior probability of pred1
    given pred2 is true, according to the formula
    
    Pos_Pr = Pr(pred2, pred1) * Pr(pred1) / Pr(pred2)
    Pr = prior probability
    '''
    return pr(pred2, pred1) * pr(pred1) / pr(pred2)

# Cell 3
# Question 1 part (a)

# Number of eggs in the bunch
N = sum(map(c_map, egg_list))

print(f'Number of eggs in this bunch {N = }')

# Cell 4
# Question 1 part (b)

# predicate for large eggs
f_large = lambda x: x.label == 'Large'

# prior probability of Large egg
pr_large = pr(f_large)

# predicate for jumbo eggs
f_jumbo = lambda x: x.label == 'Jumbo'

# prior probability of Jumbo egg
pr_jumbo = pr(f_jumbo)

# Rounded to 5 decimal places
print(f'Pr{{Large}} = {pr_large:.5}')
print(f'Pr{{Jumbo}} = {pr_jumbo:.5}')

# Cell 5
# Question 1 part (c)

# Large egg longer than 3 cm
# Note: strict inequality is implied

# predicate for large eggs
f_large = lambda x: x.label == 'Large'

# predicate for eggs longer than 3cm
f_gt3 = lambda x: x.feature > 3

# The likelihood
pr_large_gt3 = pr(f_gt3, f_large)

# Rounded to 5 decimal places
print(f'Pr{{x > 3 | Large}} = {pr_large_gt3:0.5}')

# Cell 6
# Question 1 part (d)

# predicate for jumbo eggs
f_jumbo = lambda x: x.label == 'Jumbo'

# predicate for eggs shorter than 4cm
f_lt4 = lambda x: x.feature < 4

# The posterior probability
pos_pr_jumbo_lt4 = pos_pr(f_jumbo, f_lt4)

# Rounded to 5 decimal places
print(f'Posterior probability, Pr{{Jumbo | x < 4}} = {pos_pr_jumbo_lt4:.5}')

# Cell 7
# Question 1 part (e)

# predicate for eggs shorter than 4cm
f_lt4 = lambda x: x.feature < 4

# prior probability of eggs shorter than 4cm
pr_lt4 = pr(f_lt4)

# Rounded to 5 decimal places
print(f'Pr{{x < 4}} = {pr_lt4:.5}')

# Cell 8
# Question 1 part (f)

# predicate for jumbo eggs shorter than 4cm
f_jumbo_lt4 = lambda x: x.label == 'Jumbo' and x.feature < 4

# prior probabilty for jumbo eggs shorter than 4cm
pr_jumbo_lt4 = pr(f_jumbo_lt4)

# Rounded to 5 decimal places
print(f'Pr{{Jumbo, x < 4}} = {pr_jumbo_lt4:.5}')

# Cell 9
# Question 1 part (g)

# predicate for jumbo eggs
f_jumbo = lambda x: x.label == 'Jumbo'

# predicate for eggs shorter than 4cm
f_lt4 = lambda x: x.feature < 4

# the likelihood of eggs shorter than 4cm given it's jumbo
pr_lt4_jumbo = pr(f_lt4, f_jumbo)

# Rounded to 5 decimal places
print(f'Pr{{x < 4 | Jumbo}} = {pr_lt4_jumbo:.5}')

# Cell 10
# Question 1 part (h)

# predicate for jumbo eggs
f_jumbo = lambda x: x.label == 'Jumbo'

# predicate for eggs shorter than 4cm
f_lt4 = lambda x: x.feature < 4

lhs = pos_pr(f_jumbo, f_lt4)
rhs = pr(f_lt4, f_jumbo) * pr(f_jumbo) / pr(f_lt4)

print(f'LHS = {lhs}')
print(f'RHS = {rhs}')

print(f'LHS = RHS => {lhs == rhs}')

# Cell 12
# Question 1 part (i)

def g(x: int) -> str:
    '''
    Params:
        | x: int | The feature to use classify the egg
    Returns:
        str: The classification of the egg using MAP
    
    This function classifies an egg of the given feature
    according to the MAP/Bayesian decision rule, where,
    
    If Pr{Large | x} > Pr{Jumbo | x}
        g(x) = Large
    Else
        g(x) = Jumbo
    
    Pr is the posterior probability
    '''
    # predicate for large eggs
    f_large = lambda e: e.label == 'Large'
    
    # predicate for jumbo eggs
    f_jumbo = lambda e: e.label == 'Jumbo'
    
    # predicate foe eggs of length x
    f_x = lambda e: e.feature == x
    
    # posterior probabilty for large eggs of length x
    pr_large_x = pos_pr(f_large, f_x)
    
    # posterior probabilty for jumbo eggs of length x
    pr_jumbo_x = pos_pr(f_jumbo, f_x)
    
    # classification according to MAP
    if pr_large_x > pr_jumbo_x:
        classification = 'Large'
    else:
        classification = 'Jumbo'
    
    return classification

for i in range(1, 6):
    print(f'g({i}) = {g(i)}')

# Cell 13
# Question 1 part (j)

for i in range(1, 6):
    classification = g(i)
    correct_filter = lambda x: x.feature == i and x.label == classification
    feature_filter = lambda x: x.feature == i
    correct = sum(map(c_map, filter(correct_filter, egg_list)))
    total = sum(map(c_map, filter(feature_filter, egg_list)))
    p_correct = correct / total
    print(f'Probability of correct classification for (x = {i}) = {p_correct}')

