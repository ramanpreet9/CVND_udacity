import matplotlib.pyplot as plt
'''
The color class creates a color from 3 values, r, g, and b (red, green, and blue).

attributes:
    r - a value between 0-255 for red
    g - a value between 0-255 for green
    b - a value between 0-255 for blue
'''
    
class Color(object):
    
    # __init__ is called when a color is constructed using color.Color(_, _, _)
    def __init__(self, r, g, b):
        # Setting the r value
        self.r = r
        self.b = b
        self.g = g
        ## TODO: Set the other two color variables g and b
        

    # __repr__ is called when a color is printed using print(some_color)
    # It must return a string
    def __repr__(self):
        '''Display a color swatch and then return a text description of r,g,b values.'''
        
        plt.imshow([[(self.r/255, self.g/255, self.b/255)]])
        
        ## TODO: Write a string representation for the color
        ## ex. "rgb = [self.r, self.g, self.b]"
        ## Right now this returns an empty string
        string = 'rgb = [{0}, {1}, {2}]'.format(self.r, self.g, self.b)
        
        return string
    
    def __add__(self, other):
        '''add r,g,b values.'''
        
        new_r = self.r + other.r
        new_b = self.b + other.b
        new_g = self.g + other.g
        
        return Color(new_r//2, new_g//2, new_b//2)