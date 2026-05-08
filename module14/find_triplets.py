from itertools import combinations

species = ["R", "P", "S", "L", "K"]

beats = {
    "R": {"S", "L"},
    "P": {"R", "K"},
    "S": {"P", "L"},
    "L": {"P", "K"},
    "K": {"R", "S"},
}

def coexists(triplet):
    for x in triplet:
        num_wins = sum(y in beats[x] for y in triplet if y != x)
        num_losses = sum(x in beats[y] for y in triplet if y != x)
        if num_wins != 1 or num_losses != 1:
            return False
    return True

triplets = []

for triplet in combinations(species, 3):
    if coexists(triplet):
        triplets.append(triplet)

print("Triplet Coexistence sets:")
for triplet in triplets:
    print(triplet)