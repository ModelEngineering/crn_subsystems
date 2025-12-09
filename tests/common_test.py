
MODEL1 = """
J1: -> S1; k1
J2: S2 -> 2 S3 + S1; k2*S2
J3: S2 -> ; k3*S2
J4: S3 -> ; k4*S2
J5: S3 -> 3 S2 + S1; k2*S3
S1 = 10
S2 = 0
S3 = 0
k1 = 1
k2 = 2
k3 = 3
k4 = 4
"""
MODEL2 = """
        model random_crn()
        species S1_, S2_, S3_, S4_, S5_;

        J1: -> S1_; k1
        J2: S5_ -> 2 S4_ + 3 S4_ + S4_ + 2 S4_ + 3 S4_; k2 * S5_
        J3: S1_ -> 3 S4_ + 2 S5_ + 3 S5_ + S3_ + S3_; k3 * S1_
        J4: S2_ -> 3 S4_ + S3_ + 2 S3_ + S4_ + 3 S5_; k4 * S2_
        J5: S1_ -> S5_; k5 * S1_
        J6: S2_ -> 3 S3_ + 2 S3_ + 2 S3_ + S5_ + S5_; k6 * S2_
        J7: S1_ -> S4_ + 2 S5_ + 2 S4_ + S3_ + S5_; k7 * S1_
        J8: S2_ -> S4_ + 2 S5_ + S3_ + 3 S3_; k8 * S2_
        J9: S2_ -> S4_ + 3 S3_; k9 * S2_
        J10: S1_ -> S5_ + 2 S4_ + S4_; k10 * S1_

        # Rate constants
        k1 = 0.9742
        k2 = 0.7012
        k3 = 0.1698
        k4 = 0.4629
        k5 = 0.5307
        k6 = 0.4652
        k7 = 0.8688
        k8 = 0.8631
        k9 = 0.7551
        k10 = 0.1057

        # Species initialization
        S1_ = 1  # Input boundary species
        S2_ = 0
        S3_ = 0
        S4_ = 0
        S5_ = 0


        # Degradation reactions
        JD1: S5_ -> ; kd_0 * S5_
        kd_0 = 3.6785
        JD2: S4_ -> ; kd_1 * S4_
        kd_1 = 12.3013
        JD3: S3_ -> ; kd_2 * S3_
        kd_2 = 11.3991
        end
""" 


MODEL3 = """
J1: S1 -> 2 S2; k1
J2: S2 + 2 S1 -> 2 S3 + S1; k2*S2*S1*S1
J3: S2 -> ; k3*S2
J4: S3 -> ; k4*S2
J5: S3 -> 3 S2 + S1; k2*S3
S1 = 10
S2 = 0
S3 = 0
k1 = 1
k2 = 2
k3 = 3
k4 = 4
"""

MODEL_SEQUENTIAL = """
J1: S1 -> S2; k1*S1
J2: S2 -> S3; k2*S2
J3: S3 -> S4; k3*S3
J4: S4 -> ; k4*S4
S1 = 10
S2 = 0
S3 = 0
k1 = 1
k2 = 2
k3 = 3
k4 = 4
"""