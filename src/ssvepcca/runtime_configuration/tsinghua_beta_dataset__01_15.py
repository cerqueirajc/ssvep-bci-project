PARAMS = {
    "num_subjects": 15,
    "num_blocks": 4,
    "num_targets": 40,
    "num_samples": 750,
    "num_electrodes": 64,
    "sample_frequency": 250,
    "subjects": [f"S{i + 1}" for i in range(15)],
    "target_frequencies": [
         8.6,  8.8,  9. ,  9.2,  9.4,  9.6,  9.8, 10. , 10.2, 10.4, 10.6,
        10.8, 11. , 11.2, 11.4, 11.6, 11.8, 12. , 12.2, 12.4, 12.6, 12.8,
        13. , 13.2, 13.4, 13.6, 13.8, 14. , 14.2, 14.4, 14.6, 14.8, 15. ,
        15.2, 15.4, 15.6, 15.8,  8. ,  8.2,  8.4
    ],
    "target_phases": [
        4.71238898, 0.        , 1.57079633, 3.14159265, 4.71238898,
        0.        , 1.57079633, 3.14159265, 4.71238898, 0.        ,
        1.57079633, 3.14159265, 4.71238898, 0.        , 1.57079633,
        3.14159265, 4.71238898, 0.        , 1.57079633, 3.14159265,
        4.71238898, 0.        , 1.57079633, 3.14159265, 4.71238898,
        0.        , 1.57079633, 3.14159265, 4.71238898, 0.        ,
        1.57079633, 3.14159265, 4.71238898, 0.        , 1.57079633,
        3.14159265, 4.71238898, 0.        , 1.57079633, 3.14159265
    ],
    "electrodes": {
        "FP1": 0,
        "FPZ": 1,
        "FP2": 2,
        "AF3": 3,
        "AF4": 4,
        "F7": 5,
        "F5": 6,
        "F3": 7,
        "F1": 8,
        "FZ": 9,
        "F2": 10,
        "F4": 11,
        "F6": 12,
        "F8": 13,
        "FT7": 14,
        "FC5": 15,
        "FC3": 16,
        "FC1": 17,
        "FCz": 18,
        "FC2": 19,
        "FC4": 20,
        "FC6": 21,
        "FT8": 22,
        "T7": 23,
        "C5": 24,
        "C3": 25,
        "C1": 26,
        "Cz": 27,
        "C2": 28,
        "C4": 29,
        "C6": 30,
        "T8": 31,
        "M1": 32,
        "TP7": 33,
        "CP5": 34,
        "CP3": 35,
        "CP1": 36,
        "CPZ": 37,
        "CP2": 38,
        "CP4": 39,
        "CP6": 40,
        "TP8": 41,
        "M2": 42,
        "P7": 43,
        "P5": 44,
        "P3": 45,
        "P1": 46,
        "PZ": 47,
        "P2": 48,
        "P4": 49,
        "P6": 50,
        "P8": 51,
        "PO7": 52,
        "PO5": 53,
        "PO3": 54,
        "POz": 55,
        "PO4": 56,
        "PO6": 57,
        "PO8": 58,
        "CB1": 59,
        "O1": 60,
        "Oz": 61,
        "O2": 62,
        "CB2": 63,
    }
}