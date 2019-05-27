

import os
import glob
import pandas as pd
import argparse
import xml.etree.ElementTree as ET
import cv2

def xml_to_csv(path):
    xml_list = []
    idx=0
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            fn=root.find('filename').text
            img=cv2.imread(path+"/"+fn)
            xmin=int(member[4][0].text)-10
            xmax=int(member[4][2].text)+10
            ymin=int(member[4][1].text)-10
            ymax=int(member[4][3].text)+10
            croped=img[ymin:ymax, xmin:xmax]
            cv2.imwrite("crop/"+str(idx)+fn,croped)
            idx+=1


def main():
    # Initiate argument parser
    parser = argparse.ArgumentParser(
        description="Sample TensorFlow XML-to-CSV converter")
    parser.add_argument("-i",
                        "--inputDir",
                        help="Path to the folder where the input .xml files are stored",
                        type=str)
    parser.add_argument("-o",
                        "--outputFile",
                        help="Name of output directory", type=str)
    args = parser.parse_args()

    if(args.inputDir is None):
        args.inputDir = os.getcwd()
    #if(args.outputFile is None):
    #    args.outputFile = args.inputDir + "/labels.csv"

    assert(os.path.isdir(args.inputDir))

    xml_df = xml_to_csv(args.inputDir)
    #xml_df.to_csv(args.outputFile, index=None)
    print('Successfully')


if __name__ == '__main__':
    main()
