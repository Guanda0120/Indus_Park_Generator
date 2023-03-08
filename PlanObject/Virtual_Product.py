import numpy as np


class VirtualProduct:

    """
        Define the product
    """

    def __init__(self, info_list: list):
        """
        Data Structure like CONFIG --> product list
        :param info_list: list of info
        """
        self.total_area = info_list[0]      # total area of the product
        self.floor_num = info_list[1]       # num of floor
        self.product_num = info_list[2]     # num of product
        self.sl_ratio = info_list[3]        # short to long ratio
        self.floor_height = info_list[4]    # every floor height

        # Total Height
        self.height = self.floor_height * self.floor_num
        # Half fair spacing
        if self.height <= 24:
            self.half_fair_spacing = 3
        else:
            self.half_fair_spacing = 6.5
        # The area on the first floor
        self.floor_area = self.total_area / self.floor_num
        self.long_side = np.sqrt(self.floor_area * self.sl_ratio ** 2)
        self.short_side = self.long_side * self.sl_ratio

    def __str__(self):
        return f"Total Area: {self.total_area}, Floor Num: {self.floor_num}, Product Num: {self.product_num}, Floor " \
               f"Area: {self.floor_area}, Long & Short: {self.long_side, self.short_side}, " \
               f"Total Height: {self.height}, Half Fair Spacing: {self.half_fair_spacing} "


if __name__ == "__main__":
    from CONFIG import Para

    print(VirtualProduct(Para["product_dict"]["product_1"]))
