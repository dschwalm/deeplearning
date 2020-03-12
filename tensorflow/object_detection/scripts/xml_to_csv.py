import os, sys
import glob
import pandas as pd
import xml.etree.ElementTree as ET
import argparse


def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-imageRoot',help='the folder containing the training images in test and train folders',type=str,required=True)
    args = parser.parse_args()
    
    print (args.imageRoot)
    
    for folder in ['train','test']:
        image_path = os.path.join(args.imageRoot, folder )
        xml_df = xml_to_csv(image_path)
        xml_df.to_csv(os.path.join(args.imageRoot, folder + '_labels.csv'), index=None)
    print('Successfully converted xml to csv.')


if __name__ == '__main__':
    sys.exit(main())
