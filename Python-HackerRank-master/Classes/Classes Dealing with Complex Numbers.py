# Python has built-in support for complex number objects: https://docs.python.org/3/c-api/complex.html
class Complex(complex): # Subclass
    # The traditional operator overload
    def mod(self):
        return Complex(complex.__abs__(self))
    
    def __str__(self):
        return '{0.real:.2f}{0.imag:+.2f}i'.format(self) # display in A.00+B.00i format
    
    def __add__(self, number):
        return Complex(complex.__add__(self, number))
    
    def __sub__(self, number):
        return Complex(complex.__sub__(self, number))
    
    def __mul__(self, number):
        return Complex(complex.__mul__(self, number))
    
    def __truediv__(self, number): # Python3 uses truediv as the name of the '/' operator, whereas python2 just calls this 'div'
        return Complex(complex.__truediv__(self, number))
    
#####################################################################
##                                                                 ##
##                           2nd METHOD                            ##
##                                                                 ##
#####################################################################
class Complex(object):
    def __init__(self, real, image):
        super().__init__()
        self.real=real
        self.image=image
        
    def __add__(self, no):
        return str(Complex(self.real + no.real, self.image + no.image))
        
    def __sub__(self, no):
        return str(Complex(self.real - no.real, self.image - no.image))
        
    def __mul__(self, no):
        return str(Complex((self.real * no.real) - (self.image * no.image) ,
                 (self.real * no.image) + (self.image * no.real)))

    def __truediv__(self, no):
        return str(Complex(((self.real * no.real) + (self.image * no.image)) / (no.image**2 + no.real**2),((self.image * no.real) - (self.real * no.image) ) / (no.image**2 + no.real**2)))

    def mod(self):
        return str(Complex((self.real**2 + self.image**2)**0.5,0))

    def __str__(self):
        if self.image == 0:
            result = "%.2f+0.00i" % (self.real)
        elif self.real == 0:
            if self.image >= 0:
                result = "0.00+%.2fi" % (self.image)
            else:
                result = "0.00-%.2fi" % (abs(self.image))
        elif self.image > 0:
            result = "%.2f+%.2fi" % (self.real, self.image)
        else:
            result = "%.2f-%.2fi" % (self.real, abs(self.image))
        return result
    
    
