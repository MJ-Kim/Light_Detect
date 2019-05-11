import os, sys
import xml.etree.cElementTree as ET
from PIL import Image


ratio = 1

def adjust(node):
    global ratio
    
    origin = int(node.text)
    value = int(origin/ ratio)
    node.text = str(value)

def ModifyXML(currDirName, targetFileName, targetFullFileName,xmlFullFileName):

    tree = ET.parse(targetFullFileName)
    root = tree.getroot()
    annotation = root[0]

    size = root[4]
    width_node = size[0]
    height_node = size[1]

    object_node = root[6]
    bndbox = object_node[4]
    xmin_node = bndbox[0]
    ymin_node = bndbox[1]
    xmax_node = bndbox[2]
    ymax_node = bndbox[3]


    adjust(width_node)
    adjust(height_node)
    adjust(xmin_node)
    adjust(ymin_node)
    adjust(xmax_node)
    adjust(ymax_node)

        
    # save
    tree = ET.ElementTree(root)
    tree.write(xmlFullFileName)

def Search(dirname):
    filenameList = os.listdir(dirname)
    length = len(filenameList) - 1
    
    filenames = enumerate(filenameList)    
    
    for i, filename in filenames:
        if i == length:
            break
        
        fullFileName = os.path.join(dirname, filename)
        ext = os.path.splitext(fullFileName)[-1]

        xmlDirName = os.path.join(dirname, 'xlms')
        xmlFileName = os.path.splitext(filename)[-2] + '.xml'
        xmlFullFileName = os.path.join(xmlDirName, xmlFileName)

        print('processing...', (i+1), '/', length, '(' + xmlFullFileName + ')')
        
        if ext == '.xml':
            ModifyXML(dirname, filename, fullFileName, xmlFullFileName)


def MakeXmlDir(targetDirName):
    xmlDirName = os.path.join(targetDirName, 'xlms')
    if not os.path.isdir(xmlDirName):
        os.mkdir(xmlDirName)



if __name__ == "__main__" :
    targetDirName = sys.argv[1]
    ratio = int(sys.argv[2])
    
    MakeXmlDir(targetDirName)
    Search(targetDirName)




