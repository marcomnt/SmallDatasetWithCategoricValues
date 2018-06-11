from random import uniform
import numpy as np
import pandas as pd

def mega_trend_diffusion(df, variance):
    _max = df.max().values[-1]
    _min = df.min().values[-1]
    u_set = (_min + _max)/2.0
    N_L = df[df[df.columns[-1]] < u_set].shape[0]
    N_U = df[df[df.columns[-1]] > u_set].shape[0]
    skew_L = N_L/float(N_L + N_U)
    skew_U = N_U/float(N_L + N_U)
    U = u_set + (skew_U * np.sqrt(-2.0 * (variance/N_U) * np.log(10**-20)))
    L = u_set - (skew_L * np.sqrt(-2.0 * (variance/N_L) * np.log(10**-20)))
    return (u_set, U, L)

def membership_function(x, L, U, C):
    if (L <= x and x <= C):
        return float(float(x - L)/(C - L))
    elif (C < x and x <= U):
        return float(float(U - x)/(U - C))
    return 0.0

def possibility_assessment_mechanism(threshold, mf):
    y = 0.0
    while y <= threshold:
        vx = uniform(mf['L'], mf['U'])
        y = membership_function(vx, mf['L'], mf['U'], mf['u_set'])
    return vx

def main():
    # df = pd.read_csv('datasets/automobile.data')
    # categorical_attributes_columns = [2, 3, 4, 5, 6, 7, 8, 14, 15, 17]
    df = pd.read_csv('../datasets/test.csv')
    categorical_attributes_columns = [0, 1]
    COB = []
    mtd_dict = {}
    pam_threshold = 0.7
    m = 10
    alpha = 0.5

    """Fuzzy relation extraction"""
    for i, attribute_column in enumerate(categorical_attributes_columns):
        df_groupby = df.groupby(df.columns[attribute_column])
        categories_dict = df_groupby.indices
        for category in categories_dict:
            group_df = df_groupby.get_group(category)
            group_df_variance = group_df.var()
            u_set, U, L = mega_trend_diffusion(group_df, group_df_variance)
            mtd_dict.update({category: {'u_set': u_set,
                                        'U': np.array(U),
                                        'L': np.array(L)}})
        """Generating COB matrix"""
        if i == 0:
            COB = categories_dict.keys()
        elif i == 1:
            COB = [[x1]+[x2] for x1 in COB for x2 in categories_dict.keys()]
        else:
            COB = [x1+[x2] for x1 in COB for x2 in categories_dict.keys()]
    COB = np.array(COB)

    """Sample generation"""
    df_variance = df.var()
    u_set, U, L = mega_trend_diffusion(df, df_variance)
    mtd_Y = {'u_set': u_set, 'U': np.array(U), 'L': np.array(L)}
    COB_virtual_samples = {}
    for i in range(len(COB)):
        m_virtual_outputs = []
        for j in range(m):
            m_virtual_outputs.append(
                possibility_assessment_mechanism(pam_threshold, mtd_Y))
        COB_virtual_samples.update({i: m_virtual_outputs})

    """Obtaining possibility values"""
    COB_possibility_values = {}
    for i in range(len(COB)):
        possibilities = []
        for j in range(m):
            possibility = 1.0
            for category in COB[i]:
                possibility *= membership_function(COB_virtual_samples[i][j],
                                                   mtd_dict[category]['L'],
                                                   mtd_dict[category]['U'],
                                                   mtd_dict[category]['u_set'])
            possibilities.append(possibility)
        COB_possibility_values.update({i: possibilities})

    """Alpha cut"""
    for i in range(len(COB)):
        indices = np.nonzero(np.array(COB_possibility_values[i]) > alpha)[0]
        for idx in indices:
            print COB[i], COB_virtual_samples[i][idx]

if __name__ == '__main__':
    main()
