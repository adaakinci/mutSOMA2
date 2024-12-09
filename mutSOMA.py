import numpy as np
import scipy.optimize as opt
from itertools import product


def mutSoma(pedigree_data, prob_aa, prob_cc, prob_tt, prob_gg, prop_het, num_starts, output_dir, output_name):

    pedigree_data = np.loadtxt("/Users/adaakinci/Desktop/makeVCFpedigree_output.txt", delimiter='\t', skiprows=1)
    pedigree_data = np.array(pedigree_data)

    allow_negative_intercept = "no"

    def calculate_divergence(pedigree_data, prob_aa, prob_cc, prob_tt, prob_gg, parameters):
        prob_AA = prob_aa
        prob_CC = prob_cc
        prob_TT = prob_tt
        prob_GG = prob_gg

        mutation_rate = parameters[0]

        initial_state_probs = np.array([prob_AA, prob_CC, prob_TT, prob_GG, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        # 16 different genotypes at each locus
        homo_genotypes = ["AA", "CC", "TT", "GG"]
        hetero_genotypes = ["AC", "AT", "AG", "CA", "CT", "CG", "TA", "TC", "TG", "GA", "GC", "GT"]

        # constructing matrix T1 (homo to homo)

        mat_T1 = [
            [
                (1 - mutation_rate) ** 2 if allele_1 == allele_2 else mutation_rate ** 2 / 9
                for allele_2 in homo_genotypes
            ]
            for allele_1 in homo_genotypes
        ]

        # constructing matrix T2 (homo to hetero)
        mat_T2 = [
            [
                (1 - mutation_rate) ** 2 if allele_1 == allele_2 else
                (1 - mutation_rate) * mutation_rate / 3 if (allele_1[0] == allele_2[0] or allele_1[1] == allele_2[1]) else
                mutation_rate ** 2 / 9
                for allele_2 in hetero_genotypes
            ]
            for allele_1 in homo_genotypes
        ]

        # constructing matrix T3 (hetero to homo)
        mat_T3 = [
            [
                (1 - mutation_rate) ** 2 if allele_1 == allele_2 else
                (1 - mutation_rate) * mutation_rate / 3 if (allele_1[0] == allele_2[0] or allele_1[1] == allele_2[1]) else
                mutation_rate ** 2 / 9
                for allele_2 in homo_genotypes
            ]
            for allele_1 in hetero_genotypes
        ]

        # constructing matrix T4 (hetero to hetero)
        mat_T4 = [
            [
                (1 - mutation_rate) ** 2 if allele_1 == allele_2 else
                (1 - mutation_rate) * mutation_rate / 3 if (allele_1[0] == allele_2[0] or allele_1[1] == allele_2[1]) else
                mutation_rate ** 2 / 9
                for allele_2 in hetero_genotypes
            ]
            for allele_1 in hetero_genotypes
        ]

        # constructing matrix T1_T2 (combining matrix T1 and T2 horizontally)
        mat_T1_T2 = []

        for i in range(len(mat_T1)):
            combine_T1_T2 = mat_T1[i] + mat_T2[i]
            mat_T1_T2.append(combine_T1_T2)

        # constructing matrix T3_T4 (combining matrix T3 and T4 horizontally)
        mat_T3_T4 = []

        for i in range(len(mat_T3)):
            combine_T3_T4 = mat_T3[i] + mat_T4[i]
            mat_T3_T4.append(combine_T3_T4)

        # constructing matrix G (combining T1_T2 and T3_T4 vertically)
        mat_G = mat_T1_T2 + mat_T3_T4
        mat_G = np.array(mat_G)

        #Divergence effects between samples i and j
        divergence_effects = np.array([
            [0, 1, 1, 1, 0.5, 0.5, 0.5, 0.5, 1, 1, 0.5, 1, 1, 0.5, 1, 1],
            [1, 0, 1, 1, 0.5, 1, 1, 0.5, 1, 1, 1, 0.5, 1, 1, 0.5, 1],
            [1, 1, 0, 1, 1, 0.5, 1, 1, 0.5, 1, 1, 1, 0.5, 1, 1, 1],
            [1, 1, 1, 0, 1, 1, 0.5, 1, 1, 0.5, 1, 1, 1, 0.5, 1, 1],
            [0.5, 0.5, 1, 1, 0, 1, 1, 1, 0.5, 1, 1, 1, 0.5, 0.5, 1, 0.5],
            [0.5, 1, 0.5, 1, 1, 0, 1, 1, 1, 1, 0.5, 1, 1, 1, 0.5, 0.5],
            [0.5, 1, 1, 0.5, 1, 1, 0, 1, 1, 1, 1, 0.5, 1, 1, 0.5, 1],
            [0.5, 0.5, 1, 1, 1, 1, 1, 0, 1, 0.5, 1, 1, 0.5, 1, 1, 0.5],
            [1, 1, 0.5, 1, 0.5, 1, 1, 1, 0, 1, 1, 1, 0.5, 0.5, 0.5, 1],
            [1, 1, 1, 0.5, 1, 1, 1, 0.5, 1, 0, 0.5, 1, 1, 1, 1, 1],
            [0.5, 1, 1, 1, 1, 0.5, 1, 1, 1, 0.5, 0, 0.5, 1, 0.5, 1, 1],
            [1, 0.5, 1, 1, 1, 1, 0.5, 1, 1, 1, 0.5, 0, 1, 0.5, 1, 1],
            [1, 1, 0.5, 1, 0.5, 1, 1, 0.5, 0.5, 1, 1, 1, 0, 1, 0.5, 1],
            [0.5, 1, 1, 0.5, 0.5, 1, 1, 1, 0.5, 1, 0.5, 0.5, 1, 0, 1, 0.5],
            [1, 0.5, 1, 1, 1, 0.5, 0.5, 1, 1, 1, 1, 1, 0.5, 1, 0, 1],
            [1, 1, 1, 1, 0.5, 0.5, 1, 0.5, 1, 1, 1, 1, 1, 0.5, 1, 0]
        ])

        # a list to store divergence values between sample pairs
        divergence_values = []

        # ---------------------MARKOV CHAIN CALCULATION------------------------------------------------------------------------------------------------------

        for time_columns in range(len(pedigree_data)):
            time_0 = int(pedigree_data[time_columns][0])     # time0 column of pedigree_data
            time_1 = int(pedigree_data[time_columns][1])     # time1 column of pedigree_data
            time_2 = int(pedigree_data[time_columns][2])    # time2 column of pedigree_data

            # list to hold probabilities for being in a specific genotype at time0 (1x16)
            prob_generation0 = np.dot(initial_state_probs, np.linalg.matrix_power(mat_G, time_0))
            prob_generation1 = []   # list to hold probabilities for reaching each genotype at time1
            prob_generation2 = []   # list to hold probabilities for reaching each genotype at time2

            time_diff_1 = max(0, time_1 - time_0)
            time_diff_2 = max(0, time_2 - time_0)

            for i in range(16):
                # accessing i-th row after markov chain for time_diff_1
                prob_generation1_i = np.dot(np.eye(16)[i, :], np.linalg.matrix_power(mat_G, time_diff_1))
                # accessing i-th row after markov chain for time_diff_2
                prob_generation2_i = np.dot(np.eye(16)[i, :], np.linalg.matrix_power(mat_G, time_diff_2))


                prob_generation1.append(prob_generation1_i)
                prob_generation2.append(prob_generation2_i)

            prob_generation1 = np.array(prob_generation1)
            prob_generation2 = np.array(prob_generation2)

            div_prob_t1t2 = []

            for j in range(16):
                # Probability of observing each genotype at time1, given a specific starting genotype j at time0
                genotype_in_t1 = prob_generation1[j]
                # Probability of observing each genotype at time2, given a specific starting genotype j at time0
                genotype_in_t2 = prob_generation2[j]

                # Likelihood of observing each possible pair of genotypes at two different time points (time1 and time2)
                # assuming they evolved independently from the same starting genotype at time0
                joint_prob = np.outer(genotype_in_t1, genotype_in_t2)

                # Measure of how much genetic variation is expected to develop over a specified time interval
                weighted_divergence = prob_generation0[j] * np.sum(divergence_effects * joint_prob)

                div_prob_t1t2.append(weighted_divergence)

            # Total expected divergence for the time interval between time1 and time2
            divergence_sum = sum(div_prob_t1t2)

            # List containing the divergence sum for each row in pedigree data
            divergence_values = np.append(divergence_values, divergence_sum)
            divergence_values = np.array(divergence_values)

        return divergence_values

    #---------------------LSE------------------------------------------------------------------------------------------------------

    def LSE(param):
        print(f"Current parameters: {param}")
        predicted_divergence = calculate_divergence(pedigree_data, prob_aa, prob_cc, prob_tt, prob_gg, param)
        intercept_lse = param[1]
        lse_value = np.sum((pedigree_data[:, 3] - intercept_lse - predicted_divergence) ** 2)
        print(f"LSE value: {lse_value}")
        return lse_value

    #-----------------NELDER-MEAD-------------------------------------------------------------------------------------------------------------
    # Optimization process to estimate the best mutation rate and intercept that minimize the LSE

    # Storing results of each optimization run
    optimization_results = []
    optimization_method = "Nelder-Mead"

    for start in range(num_starts):
        mutation_rate_random_start = 10 ** np.random.uniform(np.log10(1e-6), np.log10(1e-3))
        intercept_random_start = np.random.uniform(0, np.max(pedigree_data[:, 3]))
        initial_parameters = [mutation_rate_random_start, intercept_random_start]

        #result.x[0] --> optimized mutation rate, result.x[1] --> optimized intercept, result.fun --> minimized LSE value

        result = opt.minimize(LSE, initial_parameters, method=optimization_method)

        optimization_results.append({"gamma": result.x[0], "intercept": result.x[1], "success": result.success,
                                     "initial_gamma": mutation_rate_random_start, "initial_intercept": intercept_random_start, "LSE": result.fun})

    # Finding the best optimization result (lowest LSE)
    # Sorting the third element of each tuple, which is the minimized LSE value
    optimization_results = sorted(optimization_results, key=lambda x: x["LSE"])

    # List of indices for optimization results that meet the specified conditions
    index_1 = []

    # Filtering out invalid results based on intercept
    # If negative intercept allowed --> filter based on positive mutation rate and success
    if allow_negative_intercept == "yes":
        for i, row in enumerate(optimization_results):
            if row["gamma"] > 0 and row["success"]:
                index_1.append(i)
    # If negative intercept NOT allowed --> filter based on positive mutation rate, positive intercept, and success
    else:
        for i, row in enumerate(optimization_results):
            if row["gamma"] > 0 and row["intercept"] > 0 and row["success"]:
                index_1.append(i)

    # List of indices of all remaining optimization results that don't satisfy the conditions for index_1
    index_2 = list(set(range(len(optimization_results))) - set(index_1))

    # List of valid optimization results
    optimization_results_1 = []
    # List of invalid or less reliable optimization results
    optimization_results_2 = []

    for i in index_1:
        optimization_results_1.append(optimization_results[i])

    for i in index_2:
        optimization_results_2.append(optimization_results[i])

    #-----------------CALCULATION WITH OPTIMAL MUTATION RATES AND INTERCEPT-------------------------------------------------------------------------------------------------------------

    p0aa = prob_aa * (1 - prop_het)
    p0cc = prob_cc * (1 - prop_het)
    p0tt = prob_tt * (1 - prop_het)
    p0gg = prob_gg * (1 - prop_het)
    p0ac = prop_het * 1/12
    p0at = prop_het * 1/12
    p0ag = prop_het * 1/12
    p0ca = prop_het * 1/12
    p0ct = prop_het * 1/12
    p0cg = prop_het * 1/12
    p0ta = prop_het * 1/12
    p0tc = prop_het * 1/12
    p0tg = prop_het * 1/12
    p0ga = prop_het * 1/12
    p0gc = prop_het * 1/12
    p0gt = prop_het * 1/12

    total_sum = np.sum([p0aa, p0cc, p0tt, p0gg, p0ac, p0at, p0ag, p0ca, p0ct, p0cg, p0ta, p0tc, p0tg, p0ga, p0gc, p0gt])

    if not np.isclose(total_sum, 1.0, atol=1e-9):
        raise ValueError("The initial state probabilities don't sum up to 1")

    g = optimization_results_1[0]["gamma"]
    intercept = optimization_results_1[0]["intercept"]

    prob_aa_2 = p0aa
    prob_cc_2 = p0cc
    prob_tt_2 = p0tt
    prob_gg_2 = p0gg
    prob_ac_2 = p0ac
    prob_at_2 = p0at
    prob_ag_2 = p0ag
    prob_ca_2 = p0ca
    prob_ct_2 = p0ct
    prob_cg_2 = p0cg
    prob_ta_2 = p0ta
    prob_tc_2 = p0tc
    prob_tg_2 = p0tg
    prob_ga_2 = p0ga
    prob_gc_2 = p0gc
    prob_gt_2 = p0gt

    initial_state_probs_2 = np.array([prob_aa_2, prob_cc_2, prob_tt_2, prob_gg_2, prob_ac_2, prob_at_2, prob_ag_2,
                                      prob_ca_2, prob_ct_2, prob_cg_2, prob_ta_2, prob_tc_2, prob_tg_2, prob_ga_2,
                                      prob_gc_2, prob_gt_2])

    homo_genotypes = ["AA", "CC", "TT", "GG"]
    hetero_genotypes = ["AC", "AT", "AG", "CA", "CT", "CG", "TA", "TC", "TG", "GA", "GC", "GT"]

    mat_T1_new = [
        [
            (1 - g) ** 2 if allele_1 == allele_2 else g ** 2 / 9
            for allele_2 in homo_genotypes
        ]
        for allele_1 in homo_genotypes
    ]

    mat_T2_new = [
        [
            (1 - g) ** 2 if allele_1 == allele_2 else
            (1 - g) * g / 3 if (allele_1[0] == allele_2[0] or allele_1[1] == allele_2[1]) else
            g ** 2 / 9
            for allele_2 in hetero_genotypes
        ]
        for allele_1 in homo_genotypes
    ]

    mat_T3_new = [
        [
            (1 - g) ** 2 if allele_1 == allele_2 else
            (1 - g) * g / 3 if (allele_1[0] == allele_2[0] or allele_1[1] == allele_2[1]) else
            g ** 2 / 9
            for allele_2 in homo_genotypes
        ]
        for allele_1 in hetero_genotypes
    ]

    mat_T4_new = [
        [
            (1 - g) ** 2 if allele_1 == allele_2 else
            (1 - g) * g / 3 if (allele_1[0] == allele_2[0] or allele_1[1] == allele_2[1]) else
            g ** 2 / 9
            for allele_2 in hetero_genotypes
        ]
        for allele_1 in hetero_genotypes
    ]

    mat_T1_T2_new = []

    for i in range(len(mat_T1_new)):
        combine_T1_T2 = mat_T1_new[i] + mat_T2_new[i]
        mat_T1_T2_new.append(combine_T1_T2)

    mat_T3_T4_new = []

    for i in range(len(mat_T3_new)):
        combine_T3_T4 = mat_T3_new[i] + mat_T4_new[i]
        mat_T3_T4_new.append(combine_T3_T4)

    mat_G = mat_T1_T2_new + mat_T3_T4_new
    mat_G_new = np.array(mat_G)

    divergence_effects = np.array([
        [0, 1, 1, 1, 0.5, 0.5, 0.5, 0.5, 1, 1, 0.5, 1, 1, 0.5, 1, 1],
        [1, 0, 1, 1, 0.5, 1, 1, 0.5, 1, 1, 1, 0.5, 1, 1, 0.5, 1],
        [1, 1, 0, 1, 1, 0.5, 1, 1, 0.5, 1, 1, 1, 0.5, 1, 1, 1],
        [1, 1, 1, 0, 1, 1, 0.5, 1, 1, 0.5, 1, 1, 1, 0.5, 1, 1],
        [0.5, 0.5, 1, 1, 0, 1, 1, 1, 0.5, 1, 1, 1, 0.5, 0.5, 1, 0.5],
        [0.5, 1, 0.5, 1, 1, 0, 1, 1, 1, 1, 0.5, 1, 1, 1, 0.5, 0.5],
        [0.5, 1, 1, 0.5, 1, 1, 0, 1, 1, 1, 1, 0.5, 1, 1, 0.5, 1],
        [0.5, 0.5, 1, 1, 1, 1, 1, 0, 1, 0.5, 1, 1, 0.5, 1, 1, 0.5],
        [1, 1, 0.5, 1, 0.5, 1, 1, 1, 0, 1, 1, 1, 0.5, 0.5, 0.5, 1],
        [1, 1, 1, 0.5, 1, 1, 1, 0.5, 1, 0, 0.5, 1, 1, 1, 1, 1],
        [0.5, 1, 1, 1, 1, 0.5, 1, 1, 1, 0.5, 0, 0.5, 1, 0.5, 1, 1],
        [1, 0.5, 1, 1, 1, 1, 0.5, 1, 1, 1, 0.5, 0, 1, 0.5, 1, 1],
        [1, 1, 0.5, 1, 0.5, 1, 1, 0.5, 0.5, 1, 1, 1, 0, 1, 0.5, 1],
        [0.5, 1, 1, 0.5, 0.5, 1, 1, 1, 0.5, 1, 0.5, 0.5, 1, 0, 1, 0.5],
        [1, 0.5, 1, 1, 1, 0.5, 0.5, 1, 1, 1, 1, 1, 0.5, 1, 0, 1],
        [1, 1, 1, 1, 0.5, 0.5, 1, 0.5, 1, 1, 1, 1, 1, 0.5, 1, 0]
    ])

    divergence_values_2 = []

    for time_columns in range(len(pedigree_data)):
        time_0_new = int(pedigree_data[time_columns][0])  # time0 column of pedigree_data
        time_1_new = int(pedigree_data[time_columns][1])  # time1 column of pedigree_data
        time_2_new = int(pedigree_data[time_columns][2])  # time2 column of pedigree_data

        # list to hold probabilities for being in a specific genotype at time0 (1x16)
        prob_generation0 = np.dot(initial_state_probs_2, np.linalg.matrix_power(mat_G_new, time_0_new))
        prob_generation1 = []  # list to hold probabilities for reaching each genotype at time1
        prob_generation2 = []  # list to hold probabilities for reaching each genotype at time2

        time_diff_1_new = time_1_new - time_0_new
        time_diff_2_new = time_2_new - time_0_new

        for i in range(16):
            # accessing i-th row after markov chain for time_diff_1
            prob_generation1_i = np.dot(np.eye(16)[i, :], np.linalg.matrix_power(mat_G, time_diff_1_new))
            # accessing i-th row after markov chain for time_diff_2
            prob_generation2_i = np.dot(np.eye(16)[i, :], np.linalg.matrix_power(mat_G, time_diff_2_new))

            prob_generation1.append(prob_generation1_i)
            prob_generation2.append(prob_generation2_i)

        prob_generation1_new = np.array(prob_generation1)
        prob_generation2_new = np.array(prob_generation2)

        div_prob_t1t2_new = []

        for j in range(16):
            # Probability of observing each genotype at time1, given a specific starting genotype j at time0
            genotype_in_t1 = prob_generation1_new[j]
            # Probability of observing each genotype at time2, given a specific starting genotype j at time0
            genotype_in_t2 = prob_generation2_new[j]

            # Likelihood of observing each possible pair of genotypes at two different time points (time1 and time2)
            # assuming they evolved independently from the same starting genotype at time0
            joint_prob = np.outer(genotype_in_t1, genotype_in_t2)

            # Measure of how much genetic variation is expected to develop over a specified time interval
            weighted_divergence = prob_generation0[j] * np.sum(divergence_effects * joint_prob)

            div_prob_t1t2_new.append(weighted_divergence)

        # Total expected divergence for the time interval between time1 and time2
        divergence_sum = sum(div_prob_t1t2_new)

        # List containing the divergence sum for each row in pedigree data
        divergence_values_2.append(divergence_sum)
        #print(divergence_values_2)

    # Measure the error between observed and predicted divergence
    residual = pedigree_data[:, 3] - intercept - divergence_values_2
    # Time difference between two samples
    delta_t = pedigree_data[:, 1] + pedigree_data[:, 2] - 2 * pedigree_data[:, 0]
    # Predicted divergence
    div_pred = divergence_values_2 + intercept
    # Augmentation adds time differences and predictions to the pedigree data
    augmented_pedigree_data = np.column_stack((pedigree_data, delta_t, div_pred, residual))

    info = [["p0aa", prob_aa_2], ["p0cc", prob_cc_2], ["p0tt", prob_tt_2], ["p0gg", prob_gg_2], ["Nstarts", num_starts],
            ["prop.het", prop_het], ["optim.method", "Nelder-Mead"]]

    # Extracts the maximum time value from the second (time1) and third (time2) columns of pedigree_data
    max_time = int(np.max(pedigree_data[:, [1, 2]]))

    # These time sequences represent all possible evolutionary time points over which divergence could have occurred
    time1 = np.arange(1, max_time + 1)
    time2 = np.arange(1, max_time + 1)

    # All possible combinations of time intervals for paired samples
    time_out = np.array(list(product(time1, time2)))

    # Adds a time0 column with zeros to represent the starting point of evolution
    time0 = np.zeros((len(time_out), 1))
    pedigree_new = np.hstack((time0, time_out))

    delta_t_new = pedigree_new[:, 1] + pedigree_new[:, 2] - 2 * pedigree_new[:, 0]
    pedigree_new = np.hstack((pedigree_new, delta_t_new.reshape(-1,1)))

    # Keeps only the first occurrence of each unique delta_t
    unique_values, unique_indices = np.unique(pedigree_new[:, 3], return_index=True)
    pedigree_new = pedigree_new[unique_indices]

    # Keeps only the first three columns (time0, time1, time2)
    pedigree_new = pedigree_new[:, :3]

    divergence_values_3 = []

    for time_columns in range(len(pedigree_new)):
        time_0_new = int(pedigree_new[time_columns][0])
        time_1_new = int(pedigree_new[time_columns][1])
        time_2_new = int(pedigree_new[time_columns][2])

        prob_generation0 = np.dot(initial_state_probs_2, np.linalg.matrix_power(mat_G_new, time_0_new))
        prob_generation1 = []
        prob_generation2 = []

        time_diff_1_new = time_1_new - time_0_new
        time_diff_2_new = time_2_new - time_0_new

        for i in range(16):
            # accessing i-th row after markov chain for time_diff_1
            prob_generation1_i = np.dot(np.eye(16)[i, :], np.linalg.matrix_power(mat_G, time_diff_1_new))
            # accessing i-th row after markov chain for time_diff_2
            prob_generation2_i = np.dot(np.eye(16)[i, :], np.linalg.matrix_power(mat_G, time_diff_2_new))

            prob_generation1.append(prob_generation1_i)
            prob_generation2.append(prob_generation2_i)

        prob_generation1_new = np.array(prob_generation1)
        prob_generation2_new = np.array(prob_generation2)

        div_prob_t1t2_new_2 = []

        for j in range(16):
            genotype_in_t1_new = prob_generation1_new[j]

            genotype_in_t2_new = prob_generation2_new[j]

            joint_prob_new = np.outer(genotype_in_t1_new, genotype_in_t2_new)

            weighted_divergence_new = prob_generation0[j] * np.sum(divergence_effects * joint_prob_new)

            div_prob_t1t2_new_2.append(weighted_divergence_new)
            #print(div_prob_t1t2_new_2)

        divergence_sum_2 = sum(div_prob_t1t2_new_2)

        divergence_values_3.append(divergence_sum_2)

    div_sim = divergence_values_3 + intercept

    delta_t_new_2 = pedigree_new[:, 1] + pedigree_new[:, 2] - 2 * pedigree_new[:, 0]

    pedigree_new = np.column_stack((pedigree_new, div_sim, delta_t_new_2))

    pedigree_new = pedigree_new[pedigree_new[:, 4].argsort()]
    model = "mutSOMA.py"

    abfree_out = {"estimates": optimization_results_1, "estimates_flagged": optimization_results_2,
                  "pedigree": augmented_pedigree_data,"settings": info, "model": model, "for_fit_plot": pedigree_new}

    def recursively_clean_data(data):
        if isinstance(data, dict):
            return {key: recursively_clean_data(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [recursively_clean_data(item) for item in data]
        elif isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, np.float64):
            return float(data)
        elif isinstance(data, np.int64):
            return int(data)
        else:
            return data

    cleaned_abfree_out = recursively_clean_data(abfree_out)

    import pprint
    pprint.pprint(cleaned_abfree_out)

if __name__ == "__main__":
    mutSoma(pedigree_data="makeVCFpedigree_output.txt", prob_aa=0.25, prob_cc=0.3, prob_tt=0.15, prob_gg=0.3, prop_het=0.1, num_starts=10, output_dir="/Users/adaakinci/Desktop", output_name="mutsoma_output")