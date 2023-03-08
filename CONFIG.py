
Para = {
    "file_dir": "D:\\LiGD\\Indus_Park\\Test_File\\test_ShenYang_1.3dm",
    "layer_name": "Planning_Area",
    "max_area":  20000,
    "road_width": 7,
    "redline_dist": 5,
    # In product dictionary, key is product name, value is a list.
    # list[0] refers to total building area.
    # list[1] refers to num of floor.
    # list[2] refers to num of product.
    # list[3] short long ratio
    # list[4] floor height
    "product_dict": {
        "product_1": [16500, 15, 1, 0.6, 3],
        "product_2": [1800, 3, 34, 0.6, 3],
        "product_3": [2400, 3, 18, 0.6, 3],
        "product_4": [5050, 5, 6, 0.6, 3],
    },
    "sample_dist": 3,
    "random_seeds": 42,
    "out_file_dir": "D:\\LiGD\\Indus_Park\\OUTPUT\\Rhino_model\\generate_XKOOL_42_2w.3dm"
}

"""
Para = {
    "file_dir": "D:\\LiGD\\Indus_Park\\Test_File\\test_shenyang.3dm",
    "layer_name": "Planning_Area",
    "max_area":  30000,
    "road_width": 7,
    "redline_dist": 5,
    # In product dictionary, key is product name, value is a list.
    # list[0] refers to total building area.
    # list[1] refers to num of floor.
    # list[3] refers to num of product.
    # list[4] short long ratio
    # list[5] floor height
    "product_dict": {
        "product_1": [30000, 6, 12, 0.6, 3],
        "product_2": [12000, 4, 26, 0.6, 3],
        "product_3": [4000, 4, 36, 0.6, 3]
    },
    "sample_dist": 3,
    "random_seeds": 42,
    "out_file_dir": "D:\\LiGD\\Indus_Park\\OUTPUT\\Rhino_model\\generate_shenyang_42_3w.3dm"
}
"""
"""
Para = {
    "file_dir": "D:\\LiGD\\Indus_Park\\Test_File\\test_shenyang.3dm",
    "layer_name": "Planning_Area",
    "max_area":  20000,
    "road_width": 7,
    "redline_dist": 5,
    # In product dictionary, key is product name, value is a list.
    # list[0] refers to total building area.
    # list[1] refers to num of floor.
    # list[3] refers to num of product.
    # list[4] short long ratio
    # list[5] floor height
    "product_dict": {
        "product_1": [30000, 6, 15, 0.6, 3],
        "product_2": [12000, 4, 32, 0.6, 3],
        "product_3": [4000, 4, 45, 0.6, 3]
    },
    "sample_dist": 3,
    "random_seeds": 12,
    "out_file_dir": "D:\\LiGD\\Indus_Park\\OUTPUT\\Rhino_model\\generate_shenyang_12_2w_add.3dm"
}
"""