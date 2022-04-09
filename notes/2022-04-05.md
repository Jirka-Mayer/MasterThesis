Questions:

- what parameters to measure first? (noise params, split param, magic param)
- sup / unsup loss magic constant ratio + averaging + batch size?
- grid search?
- train to completion / stop early? Choose best model for eval?

Results:

            seed [42 43 44 45]

    sup   unsup  loss                                f1
    0.05  0.0    [0.018, 0.012, 0.020, 0.021] 0.018  [0.41, 0.74, 0.34, 0.50] 0.50
    0.05  0.1    [0.014, 0.017, 0.017, 0.019] 0.017  [0.72, 0.53, 0.61, 0.43] 0.57
    0.05  0.5    [0.015, 0.016, 0.013, 0.016] 0.015  [0.67, 0.64, 0.72, 0.65] 0.67
    0.05  0.95   [0.016, 0.014, 0.017, 0.016] 0.016  [0.58, 0.71, 0.39, 0.64] 0.58


    sup   unsup  loss            f1
    0.05  0.0    [0.015, 0.015]  [0.61, 0.55]
    0.05  0.05   [0.029, 0.026]  [0.58, 0.59]
    0.05  0.1    [0.026, 0.020]  [0.61, 0.68]
    0.05  0.5    [0.014,      ]  [0.21, ~.18]
