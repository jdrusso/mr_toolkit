import pytest
import numpy as np


@pytest.fixture
def ref_avg_bin_weight():

    return 268 / 3125


@pytest.fixture
def ref_tridiag():
    """
    Reference tridiagonal matrix.
    """

    tridiag_arr = np.array(
        [
            [4, 1, 0, 0, 0, 0],
            [1, 3, 1, 0, 0, 0],
            [0, 1, 3, 1, 0, 0],
            [0, 0, 1, 3, 1, 0],
            [0, 0, 0, 1, 3, 1],
            [0, 0, 0, 0, 1, 4],
        ],
        dtype=float,
    )

    # Return normalized array
    return tridiag_arr / np.sum(tridiag_arr, axis=1)[:, np.newaxis]

@pytest.fixture
def ref_tridiag_equil():
    """
    Equilibrium distribution for the reference matrix.
    """

    equil = np.ones(shape=6)

    return equil / 6.


@pytest.fixture
def ref_coarse_grained_uniform_weights():
    """
    Hand-calculated reference matrix under uniform initial weights. S=1, lag=1

    A tridiagonal, with 3s on the diagonal and 1 on the off-diagonal, coarse-grained with microbin weights
    [1,1,1,1,1,1].
    """

    cg_arr = np.array([[4, 1, 0, 0], [1, 8, 1, 0], [0, 1, 8, 1], [0, 0, 1, 4]])

    # This normalization is irrelevant since I need to row normalize anyways
    # return cg_arr / np.array([24, 30, 30, 24])[:, np.newaxis]

    return cg_arr / np.sum(cg_arr, axis=1)[:, np.newaxis]


@pytest.fixture
def ref_coarse_grained_nonuniform_weights():
    """
    Hand-calculated reference matrix, under initial weights of [1,2,3,4,5,6]/21. S=1, lag=1

    A tridiagonal, with 3s on the diagonal and 1 on the off-diagonal, coarse-grained with microbin weights
    [1,2,3,4,5,6].
    """

    cg_arr = np.array(
        [[4, 1, 0, 0], [2, 20, 3, 0], [0, 4, 36, 5], [0, 0, 6, 24],]
    )  # / 21, but irrelevant because renorming anyways

    return cg_arr / np.sum(cg_arr, axis=1)[:, np.newaxis]


@pytest.fixture
def ref_coarse_grained_timeavged_weights():
    """
    Hand-calculated reference matrix.
    Initial weights of [1,2,3,4,5,6]/21, time-avged with S = 10, lag = 5
    """

    cg_arr = np.array(
        [
            [
                20440181 / 615234375,
                419824 / 13671875,
                26239 / 9765625,
                26239 / 1230468750,
            ],
            [
                6370334 / 123046875,
                8562912 / 68359375,
                4113276 / 68359375,
                3418888 / 615234375,
            ],
            [
                4849862 / 615234375,
                19203922 / 205078125,
                45230014 / 205078125,
                12529666 / 123046875,
            ],
            [
                105011 / 1230468750,
                105011 / 9765625,
                1680176 / 13671875,
                81803569 / 615234375,
            ],
        ]
    )

    return cg_arr / np.sum(cg_arr, axis=1)[:, np.newaxis]


@pytest.fixture
def initial_weights():
    return np.array([1, 2, 3, 4, 5, 6]) / 21


@pytest.fixture
def ref_twostep_weights():
    """
    Reference time-avged bin weights for S=2, lag=1, calculated by hand in Mathematica.

    Sum[w0.MatrixPower[P, s], {s, 0, 3}] where P is the tridiagonal with 3s on the diagonal and 1s off.
    """
    return np.array([22 / 420, 2 / 21, 1 / 7, 4 / 21, 5 / 21, 59 / 210])  # / 2


@pytest.fixture
def ref_tenstep_weights():
    """
    Reference time-avged bin weights for S=10, lag=1, calculated by hand in Mathematica.

    Sum[w0.MatrixPower[P, s], {s, 0, 9}] where P is the tridiagonal with 3s on the diagonal and 1s off.
    """
    return (
        np.array(
            [
                4526963 / 58593750,
                7170683 / 68359375,
                29725331 / 205078125,
                38634044 / 205078125,
                46847326 / 205078125,
                5001429 / 19531250,
            ]
        )
        # / 10
    )


@pytest.fixture
def ref_tenstep_lag5_weights():
    """
    Reference time-avged bin weights for S=10, at a lag of 5, calculated by hand in Mathematica.

    Sum[w0.MatrixPower[P, s], {s, 0, 9}] where P is the tridiagonal with 3s on the diagonal and 1s off.
    """
    return np.array(
        [
            26239 / 393750,
            2794 / 28125,
            9412 / 65625,
            12463 / 65625,
            6581 / 28125,
            105011 / 393750,
        ]
    )


@pytest.fixture
def ref_cg_map():
    return [[0], [1, 2], [3, 4], [5]]


@pytest.fixture
def ref_occ_th2():
    """
    Reference occupancy matrix.

    Computed starting with the tridiagonal 3/1, initial weights [1,2,3,4,5,6]/21, s=10, and a time horizon of 2.
    """
    occ = np.array(
        [
            [2079801916 / 2806886345, 727084429 / 2806886345, 0, 0],
            [514406381 / 4594634600, 170051669 / 229731730, 679194839 / 4594634600, 0],
            [0, 230103634 / 1927513225, 570443537 / 771005290, 542601497 / 3855026450],
            [0, 0, 2472134321 / 9497801155, 7025666834 / 9497801155],
        ]
    )

    return occ / np.sum(occ, axis=1)[:, np.newaxis]


@pytest.fixture
def ref_occ_th5():
    """
    Reference occupancy matrix.

    Computed starting with the tridiagonal 3/1, initial weights [1,2,3,4,5,6]/21, and a time horizon of 5.
    """

    occ = np.array(
        [
            [
                58647686159 / 93149785705,
                33250524011 / 93149785705,
                249213069 / 18629957141,
                1102038 / 18629957141,
            ],
            [
                3326613751 / 20527245850,
                80338306663 / 127268924270,
                126431779909 / 636344621350,
                59956257 / 7486407310,
            ],
            [
                1417268631 / 217262325730,
                185134720091 / 1086311628650,
                137151868337 / 217262325730,
                208331223719 / 1086311628650,
            ],
            [
                4410462 / 67502855359,
                969049431 / 67502855359,
                122477600989 / 337514276795,
                210169376341 / 337514276795,
            ],
        ]
    )

    return occ / np.sum(occ, axis=1)[:, np.newaxis]
