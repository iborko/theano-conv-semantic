

class Sample(object):
    """
    Represents a sample from the training dataset
    """

    def __init__(self, name, image, marked_image):
        """
        name: str
            image name
        image: numpy 2d-array
            image for segmentation
        marked_image: numpy 2d-array
            semantically marked image
        """
        self.name = name
        self.image = image
        self.marked_image = marked_image
