class RotatingList:
    def __init__(self, items):
        """
        Initialize the RotatingList with a list of items and set the starting index to 0.

        :param items: List of items to rotate through.
        """
        self.items = items
        self.index = 0

    def rotate(self):
        """
        Rotate to the next item in the list. If the end of the list is reached, wrap around to the beginning.

        :return: The current item after rotation.
        """
        self.index = (self.index + 1) % len(self.items)
        return self.items[self.index]

    def get_current_item(self):
        """
        Get the current item in the list based on the current index.

        :return: The current item.
        """
        return self.items[self.index]
