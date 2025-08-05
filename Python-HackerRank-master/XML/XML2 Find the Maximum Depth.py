maxdepth = 0
def depth(elem, level): # Take a simple DFS approach
    global maxdepth
    # your code goes here
    level += 1
    [depth(element, level) for element in elem.getchildren()]
    maxdepth = level if level > maxdepth else maxdepth
    return maxdepth
    


################################################
##
##          2nd Method
##
################################################
    
import xml.etree.ElementTree as etree

maxdepth = 0
def depth(elem, level):
    global maxdepth
    # your code goes here 
    if (level == maxdepth):
        maxdepth += 1
    # recursive call to function to get the depth
    for child in elem:
        depth(child, level + 1) 
    return maxdepth


if __name__ == '__main__':
    n = int(input())
    xml = ""
    for i in range(n):
        xml =  xml + input() + "\n"
    tree = etree.ElementTree(etree.fromstring(xml))
    depth(tree.getroot(), -1)
    print(maxdepth)