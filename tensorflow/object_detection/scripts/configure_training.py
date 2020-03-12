import subprocess
import shlex
import argparse
import sys

def getLabelMap(labelMapAsStr):
    labelMap = {}
    for labelValue in [x for x in labelMapAsStr.split(',')]:
        labelMap[labelValue.split(':')[0]] = int(labelValue.split(':')[1])
    return labelMap

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-imageRoot',help='the folder containing the training images in test and train folders',type=str,required=True)
    parser.add_argument('-labelMap',help='Comma separated label map, e.g. label1:value1,label2:value2',type=str,required=True)
    parser.add_argument('-labelMapOutputFile',help='The file where the .pbtxt file will be generated based on the labelMap',type=str,required=True)
    args = parser.parse_args()
    
    subprocess.call(shlex.split(r'python xml_to_csv.py -imageRoot %s' % args.imageRoot), shell=False)
    
    for imageSet in ['train','test']:
        subprocess.call(shlex.split(r'python generate_tfrecord.py --csv_input={imageRoot}/{imageSet}_labels.csv --image_dir={imageRoot}/{imageSet} --output_path={imageRoot}/{imageSet}.record --label_map={labelMap}'.format(imageRoot=args.imageRoot, imageSet=imageSet, labelMap=args.labelMap)), shell=False)
    
    labelDict = getLabelMap(args.labelMap)
    
    with open(args.labelMapOutputFile,'wt') as outFile:
        for label, id in labelDict.items():
            outFile.write('item {\n')
            outFile.write("\tid: %s\n" % id)
            outFile.write("\tname: '%s'\n" % label)
            outFile.write('}\n')
            
    

if __name__ == '__main__':
    sys.exit(main())

