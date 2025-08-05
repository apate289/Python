def get_attr_number(node):
    # your code goes here
    # Use the iter(): https://docs.python.org/3/library/xml.etree.elementtree.html
    return sum([len(element.items()) for element in tree.iter()])

################################################
##
##          2nd Method
##
################################################
import sys
import xml.etree.ElementTree as etree

def get_attr_number(node):
    attribute_count = 0
    for element in node.iter():
        attribute_count += len(element.attrib)
    return attribute_count

if __name__ == '__main__':
    sys.stdin.readline()
    xml = sys.stdin.read()
    tree = etree.ElementTree(etree.fromstring(xml))
    root = tree.getroot()
    print(get_attr_number(root))