import xml.etree.ElementTree as ET
import os 
import numpy as np 
import pandas as pd 

def parse_xml(xml_file):
    """Parses the XML file and extracts the formatted character data.

    Args:
        xml_file (str): Path to the XML file.

    Returns:
        str: A single string in the format:
             "char1char2char3..., char1, <xmin, ymin, xmax, ymax>, char2, <xmin, ymin, xmax, ymax>, ..."
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    filename = os.path.basename(xml_file)

    data_string2 = ""
    characterCoord = []
    # Iterate through the characters
    for obj in root.iter('object'):
        character = obj.find('name').text 
        data_string2 += character 
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        
        coord = [xmin, ymin, xmax, ymax]
        characterCoord.append(character)
        characterCoord.append(f"{'-'.join(str(x) for x in coord)}")
        
    characterCoord.insert(0, data_string2)
    characterCoord.insert(1, filename)
    return characterCoord

def ExtractingParsedFiles(path): 
    ### returns a pandas dataframe of parsed xml files. 
    listOfFiles = [os.path.join(path, file)
                   for file in os.listdir(path)] # get the file content. 
    listOfFiles = sorted(listOfFiles) # sort the files.
    dataframe = pd.DataFrame() # construct empty dataframe. 
    for xml_file in listOfFiles: 
        characterCoord = parse_xml(xml_file)
        length = len(characterCoord)
        characterCoord = np.array(characterCoord).reshape(1, length)
        dataframe = pd.concat([dataframe, pd.DataFrame(characterCoord)])

    numcols = len(dataframe.columns)
    col_names = ["plate", "filename"]
    for i in range(1, (numcols // 2)):
        col_names.append(f"char{i-1}")
        col_names.append(f"coord{i-1}")
    dataframe.columns = col_names

    return dataframe

def ExtractAndSave(path, save_path): 
    dataframe = ExtractingParsedFiles(path)
    dataframe.to_csv(save_path, index=False)
    print(f"Data saved to {save_path}")

if __name__ == "__main__":
    path = "../OCR/data/LP-characters/annotations/"
    save_path = ".out/license_plates.csv"
    ExtractAndSave(path, save_path)