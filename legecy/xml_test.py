# from xml.dom import minidom
# import os 

# root = minidom.Document()
# annotation = root.createElement('annotation')
# root.appendChild(annotation)

# productChild = root.createElement('product')
# productChild.setAttribute('name', 'Geeks for Geeks')
  
# annotation.appendChild(productChild)


# xml_str = root.toprettyxml(indent ="\t") 
# with open("test.xml", "w") as f:
#     f.write(xml_str) 


'''
<?xml version="1.0" ?>
<root>
	<product name="Geeks for Geeks"/>
</root>
'''


# import xml.etree.cElementTree as ET

# root = ET.Element("root")
# doc = ET.SubElement(root, "doc")

# ET.SubElement(doc, "field1", name="blah").text = "some value1"
# ET.SubElement(doc, "field2", name="asdfasd").text = "some vlaue2"

# tree = ET.ElementTree(root)
# # xml_str = tree.toprettyxml# (indent ="\t") 
# print(tree.tostring(root,  pretty_print=True))
# # with open("test.xml", "w") as f:
# #     f.write(xml_str) 

# tree.write("test.xml")
# from lxml import etree
# root = etree.Element("root")
# root.append( etree.Element("child1") )
# child2 = etree.SubElement(root, "child2")
# child3 = etree.SubElement(root, "child3")
# print(etree.tostring(root, pretty_print=True))


from xml.etree.ElementTree import Element, SubElement, Comment
from ElementTree_pretty import prettify

top = Element('top')

comment = Comment('Generated for PyMOTW')
top.append(comment)

child = SubElement(top, 'child')
child.text = 'This child contains text.'

child_with_tail = SubElement(top, 'child_with_tail')
child_with_tail.text = 'This child has regular text.'
child_with_tail.tail = 'And "tail" text.'

child_with_entity_ref = SubElement(top, 'child_with_entity_ref')
child_with_entity_ref.text = 'This & that'

print (prettify(top))