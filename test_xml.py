#!/usr/bin/env python3

import xml.etree.ElementTree as ET
tree = ET.parse('sample.xml')
root = tree.getroot()


for child in root:
    print(child.tag, child.attrib)


print(root[0][0].text)
print(root[0][1].text)
print(root[0][2].text)
print(root[0][3].text)


for neighbor in root.iter('neighbor'):
    print(neighbor.attrib)


for country in root.findall('country'):
    rank = country.find('rank').text
    name = country.get('name')
    print(name, rank)
