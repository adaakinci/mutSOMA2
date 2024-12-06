pedigree = [[]]
with open("D-matrix_SingleCGfiltered.txt", "r") as d_mat:
    with open("pedigree.txt", "w") as out:
        for line in d_mat:
            elements = line.strip().split('\t')

            if len(elements) >= 6:
                pair1 = elements[3]
                pair2 = elements[4]
                d_value = elements[5]

            out.write(f"{pair1}\t{pair2}\t{d_value}\n")
